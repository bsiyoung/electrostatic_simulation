import time
from queue import Queue

import numpy as np
from numba import cuda


def cpu(img: np.ndarray, data: np.ndarray, conf: dict, progress_q: Queue, verbose: bool = True) -> None:
    min_value = np.min(data)
    max_value = np.max(data)
    max_abs = max(abs(min_value), abs(max_value))

    def get_color(value: float):
        color_from = np.array(conf['ref'])
        color_to = np.array(conf['max' if value > 0.0 else 'min'])

        pos = abs(value) / max_abs
        res = np.array([0, 0, 0], dtype=np.uint8)
        for i in range(3):
            res[i] = int(color_from[i] + (color_to[i] - color_from[i]) * pos)

        return res

    data_shape = data.shape
    down_sampling = img.shape[0] // data_shape[0]

    st_tm = time.time()
    for row_idx in range(data_shape[0]):
        r0 = row_idx * down_sampling
        r1 = r0 + down_sampling
        for col_idx in range(data_shape[1]):
            c0 = col_idx * down_sampling
            c1 = c0 + down_sampling
            img[r0:r1, c0:c1] = get_color(data[row_idx][col_idx])

        progress = (row_idx + 1) / data_shape[0] * 100
        el_tm = time.time() - st_tm
        est_tm = 100 / progress * el_tm

        progress_q.put({
            'task': 'potential_color',
            'progress': progress,
            'el_tm': el_tm,
            'est_tm': est_tm
        })

        if verbose is True:
            print('\rpotential_color : {:.3f}% | {:.2f}s/{:.2f}s'.format(progress, el_tm, est_tm), end='')

    el_tm = time.time() - st_tm
    print('\rpotential_color done {:.2f}s'.format(el_tm))


def gpu(img: np.ndarray, data: np.ndarray, conf: dict, progress_q: Queue, verbose: bool = True) -> None:
    st_tm = time.time()

    min_value = np.min(data)
    max_value = np.max(data)
    max_abs = max(abs(min_value), abs(max_value))

    data_shape = data.shape
    down_sampling = img.shape[0] // data_shape[0]

    n_data = data.shape[0] * data.shape[1]
    d_img = cuda.to_device(img)
    d_data = cuda.to_device(data)
    d_cnt = cuda.to_device(np.array([0], dtype=np.int32))

    kernel_s = cuda.stream()
    progress_s = cuda.stream()

    d_min_color = cuda.to_device(np.array(conf['min'], dtype=np.int32))
    d_ref_color = cuda.to_device(np.array(conf['ref'], dtype=np.int32))
    d_max_color = cuda.to_device(np.array(conf['max'], dtype=np.int32))

    n_thread_in_block = (16, 16)
    n_block_in_grid = (
        data.shape[1] // n_thread_in_block[0] + 1,
        data.shape[0] // n_thread_in_block[1] + 1
    )
    gpu_kernel[n_block_in_grid, n_thread_in_block, kernel_s](d_img, d_data, down_sampling, max_abs,
                                                             d_min_color, d_ref_color, d_max_color, d_cnt)

    h_cnt = [0]
    while h_cnt[0] != n_data:
        time.sleep(0.3)

        h_cnt[0] = d_cnt.copy_to_host(stream=progress_s)[0]

        progress = h_cnt[0] / n_data * 100
        el_tm = time.time() - st_tm
        est_tm = 100 / progress * el_tm if progress != 0 else np.NaN

        progress_q.put({
            'task': 'potential_color',
            'progress': progress,
            'el_tm': el_tm,
            'est_tm': est_tm
        })
        if verbose is True:
            print('\rpotential_color : {:.3f}% | {:.2f}s/{:.2f}s'.format(progress, el_tm, est_tm), end='')

    el_tm = time.time() - st_tm
    print('\rpotential_color done {:.2f}s'.format(el_tm))

    # cuda.synchronize()

    img[:] = d_img.copy_to_host()

    del d_max_color
    del d_ref_color
    del d_min_color
    del d_data
    del d_img


@cuda.jit('void(uint8[:,:,:], float32[:,:], int32, float32, int32[:], int32[:], int32[:], int32[:])')
def gpu_kernel(img, data, down_sampling, max_abs, min_color, ref_color, max_color, cnt):
    col_idx, row_idx = cuda.grid(2)
    if row_idx >= data.shape[0] or col_idx >= data.shape[1]:
        return

    r0 = row_idx * down_sampling
    r1 = r0 + down_sampling

    c0 = col_idx * down_sampling
    c1 = c0 + down_sampling

    color_from = ref_color
    color_to = max_color if data[row_idx][col_idx] > 0.0 else min_color

    pos = abs(data[row_idx][col_idx]) / max_abs
    for i in range(3):
        img[r0:r1, c0:c1, i] = int(color_from[i] + (color_to[i] - color_from[i]) * pos)

    cuda.atomic.add(cnt, 0, 1)
