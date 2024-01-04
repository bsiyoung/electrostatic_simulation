from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from numba import cuda


def cpu(img: np.ndarray, data: np.ndarray, conf: dict) -> None:
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


def gpu(img: np.ndarray, data: np.ndarray, conf: dict) -> None:
    scale = conf['scale']

    data_shape = data.shape
    down_sampling = img.shape[0] // data_shape[0]

    d_img = cuda.to_device(img)
    d_data = cuda.to_device(data)

    n_thread_in_block = (16, 16)
    n_block_in_grid = (
        data.shape[1] // n_thread_in_block[0] + 1,
        data.shape[0] // n_thread_in_block[1] + 1
    )
    gpu_kernel[n_block_in_grid, n_thread_in_block](d_img, d_data, scale, down_sampling)
    cuda.synchronize()

    img[:] = d_img.copy_to_host()

    del d_data
    del d_img


@cuda.jit('void(uint8[:,:,:], float32[:,:], float32, int32)')
def gpu_kernel(img, data, scale, down_sampling):
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
