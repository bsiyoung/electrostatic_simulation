from __future__ import annotations
from typing import TYPE_CHECKING

import time
from queue import Queue

import numpy as np
from numba import cuda


def cpu(img: np.ndarray, data: np.ndarray, conf: dict, progress_q: Queue, verbose: bool = True) -> None:
    scale = conf['scale']

    def chk(r, c):
        chk_res = [False, False]  # right, bottom
        base = data[r][c] // scale

        if c != data.shape[1] - 1:
            if base != (data[r][c + 1] // scale):
                chk_res[0] = True
        if r != data.shape[0] - 1:
            if base != (data[r + 1][c] // scale):
                chk_res[1] = True

        return chk_res

    data_shape = data.shape
    down_sampling = img.shape[0] // data_shape[0]

    st_tm = time.time()
    for row_idx in range(data_shape[0]):
        r0 = row_idx * down_sampling
        r1 = r0 + down_sampling
        for col_idx in range(data_shape[1]):
            c0 = col_idx * down_sampling
            c1 = c0 + down_sampling

            res = chk(row_idx, col_idx)
            if res[0] is True:
                img[r0:r1, c1 - 1:c1, :] = 0
            if res[1] is True:
                img[r1 - 1:r1, c0:c1, :] = 0

        progress = (row_idx + 1) / data_shape[0] * 100
        el_tm = time.time() - st_tm
        est_tm = 100 / progress * el_tm

        progress_q.put({
            'task': 'potential_contour',
            'progress': progress,
            'el_tm': el_tm,
            'est_tm': est_tm
        })

        if verbose is True:
            print('\rpotential_contour : {:.3f}% | {:.2f}s/{:.2f}s'.format(progress, el_tm, est_tm), end='')

    el_tm = time.time() - st_tm
    print('\rpotential_contour done {:.2f}s'.format(el_tm))


def gpu(img: np.ndarray, data: np.ndarray, conf: dict, progress_q: Queue, verbose: bool = True) -> None:
    st_tm = time.time()

    scale = conf['scale']

    data_shape = data.shape
    down_sampling = img.shape[0] // data_shape[0]

    n_data = data.shape[0] * data.shape[1]
    d_img = cuda.to_device(img)
    d_data = cuda.to_device(data)
    d_cnt = cuda.to_device(np.array([0], dtype=np.int32))

    kernel_s = cuda.stream()
    progress_s = cuda.stream()

    n_thread_in_block = (16, 16)
    n_block_in_grid = (
        data.shape[1] // n_thread_in_block[0] + 1,
        data.shape[0] // n_thread_in_block[1] + 1
    )
    gpu_kernel[n_block_in_grid, n_thread_in_block, kernel_s](d_img, d_data, scale, down_sampling, d_cnt)
    h_cnt = [0]
    while h_cnt[0] != n_data:
        time.sleep(0.3)

        h_cnt[0] = d_cnt.copy_to_host(stream=progress_s)[0]

        progress = h_cnt[0] / n_data * 100
        el_tm = time.time() - st_tm
        est_tm = 100 / progress * el_tm if progress != 0 else np.NaN

        progress_q.put({
            'task': 'potential_contour',
            'progress': progress,
            'el_tm': el_tm,
            'est_tm': est_tm
        })
        if verbose is True:
            print('\rpotential_contour : {:.3f}% | {:.2f}s/{:.2f}s'.format(progress, el_tm, est_tm), end='')

    el_tm = time.time() - st_tm
    print('\rpotential_contour done {:.2f}s'.format(el_tm))

    # cuda.synchronize()

    img[:] = d_img.copy_to_host()

    del d_data
    del d_img


@cuda.jit('void(uint8[:,:,:], float32[:,:], float32, int32, int32[:])')
def gpu_kernel(img, data, scale, down_sampling, cnt):
    col_idx, row_idx = cuda.grid(2)
    if row_idx >= data.shape[0] or col_idx >= data.shape[1]:
        return

    r0 = row_idx * down_sampling
    r1 = r0 + down_sampling
    c0 = col_idx * down_sampling
    c1 = c0 + down_sampling

    chk_right = False
    chk_bottom = False
    base = data[row_idx][col_idx] // scale
    if col_idx != data.shape[1] - 1:
        if base != (data[row_idx][col_idx + 1] // scale):
            chk_right = True
    if row_idx != data.shape[0] - 1:
        if base != (data[row_idx + 1][col_idx] // scale):
            chk_bottom = True

    if chk_right is True:
        img[r0:r1, c1 - 1:c1, :] = 0
    if chk_bottom is True:
        img[r1 - 1:r1, c0:c1, :] = 0

    cuda.atomic.add(cnt, 0, 1)
