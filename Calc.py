from __future__ import annotations

import time
from typing import Tuple, List
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from queue import Queue
    from Charge import ChargeDist

import threading

import numpy as np
from numba import cuda

import Charge


class Calc:
    def __init__(self,
                 charges: Tuple[ChargeDist],
                 phy_rect: Tuple[float, float, float, float],
                 data: np.ndarray,
                 ref_point: Tuple[float, float] | None = None,
                 device: str = 'cpu'):
        self.charges: Tuple[ChargeDist] = charges
        self.phy_rect = np.array(phy_rect, dtype=np.float32)
        self.data = data
        self.ref_point = ref_point
        self.device = device

    def do(self, progress_q: Queue, verbose=True) -> None:
        self.data.fill(0.0)

        ref_potential = 0.0
        if self.ref_point is not None:
            ref_potential = self.__get_potential(self.ref_point[0], self.ref_point[1])
        self.data -= ref_potential

        if self.device == 'cpu':
            self.do_on_cpu(progress_q, verbose=verbose)
        elif self.device == 'gpu':
            self.do_on_gpu(progress_q, verbose=verbose)

    def do_on_cpu(self, progress_q: Queue, verbose: bool = True) -> None:
        st_tm = time.time()
        lock = threading.Lock()

        n_row, n_col = self.data.shape
        n_worker = 32
        n_done_row = [0]
        th_list = []
        for worker_idx in range(n_worker):
            st_row = (n_row // n_worker) * worker_idx
            en_row = (n_row // n_worker) * (worker_idx + 1) - 1 if worker_idx != n_worker - 1 else n_row - 1
            worker = threading.Thread(target=self.cpu_worker,
                                      args=(worker_idx, self, lock, self.data, st_row, en_row, n_done_row))
            th_list.append(worker)
            worker.start()

        while n_done_row[0] != n_row:
            time.sleep(0.3)

            progress = n_done_row[0] / n_row * 100
            el_tm = time.time() - st_tm
            est_tm = 100 / progress * el_tm if progress != 0 else np.NaN

            progress_q.put({
                'task': 'calc',
                'progress': progress,
                'el_tm': el_tm,
                'est_tm': est_tm
            })

            if verbose is True:
                print('\rcalc : {:.3f}% | {:.2f}s/{:.2f}s'.format(progress, el_tm, est_tm), end='')

        el_tm = time.time() - st_tm
        print('\rcalc done {:.2f}s'.format(el_tm))

        for worker in th_list:
            worker.join()

    def do_on_gpu(self, progress_q: Queue, verbose: bool = True) -> None:
        st_tm = time.time()

        # Index
        # 0 ~ 3 : x1, y1, x2, y2
        # 4 ~ 7 : unit vec
        # 8 ~ 9 : density
        # 10 : depth
        # 11 : form
        charge_info_arr = np.zeros((len(self.charges), 12), dtype=np.float32)
        for idx, charge in enumerate(self.charges):
            buf = np.zeros(12, dtype=np.float32)
            buf[0:2] = charge.p1
            buf[2:4] = charge.p2
            buf[4:6] = charge.u_vec[0]
            buf[6:8] = charge.u_vec[1]
            buf[8] = charge.density
            buf[10] = charge.depth
            buf[11] = Charge.FORM_CONSTANT if charge.form == 'constant' else 1.0
            charge_info_arr[idx] = buf

        n_data = self.data.shape[0] * self.data.shape[1]
        d_phy_rect = cuda.to_device(self.phy_rect)
        d_data = cuda.to_device(self.data)
        d_charge = cuda.to_device(charge_info_arr)
        d_cnt = cuda.to_device(np.array([0], dtype=np.int32))

        n_thread_in_block = (16, 16)
        n_block_in_grid = (
            self.data.shape[1] // n_thread_in_block[0] + 1,
            self.data.shape[0] // n_thread_in_block[1] + 1
        )

        kernel_s = cuda.stream()
        progress_s = cuda.stream()

        gpu_kernel[n_block_in_grid, n_thread_in_block, kernel_s](d_phy_rect, d_data, d_charge, d_cnt)

        h_cnt = [0]
        while h_cnt[0] != n_data:
            time.sleep(0.3)

            h_cnt[0] = d_cnt.copy_to_host(stream=progress_s)[0]

            progress = h_cnt[0] / n_data * 100
            el_tm = time.time() - st_tm
            est_tm = 100 / progress * el_tm if progress != 0 else np.NaN

            progress_q.put({
                'task': 'calc',
                'progress': progress,
                'el_tm': el_tm,
                'est_tm': est_tm
            })
            if verbose is True:
                print('\rcalc : {:.3f}% | {:.2f}s/{:.2f}s'.format(progress, el_tm, est_tm), end='')

        el_tm = time.time() - st_tm
        print('\rcalc done {:.2f}s'.format(el_tm))

        # kernel_s.synchronize()

        self.data[:] = d_data.copy_to_host()

        del progress_s
        del kernel_s
        del d_cnt
        del d_charge
        del d_data
        del d_phy_rect

    def __get_potential(self, x: float, y: float) -> float:
        res = 0.0

        for charge in self.charges:
            res += charge.get_potential(x, y)

        return res

    def __get_sample_phy_pos(self, col_idx: int, row_idx: int):
        phy_rect = self.phy_rect
        n_row, n_col = self.data.shape

        sample_x = phy_rect[0] + (phy_rect[2] - phy_rect[0]) * col_idx / (n_col - 1)
        sample_y = phy_rect[1] - (phy_rect[1] - phy_rect[3]) * row_idx / (n_row - 1)

        return sample_x, sample_y

    @staticmethod
    def cpu_worker(th_idx: int, calc: Calc, lock: threading.Lock,
                   data: np.ndarray, st_row: int, en_row: int,
                   n_done_row: List[int]):
        buf = np.zeros((en_row - st_row + 1, data.shape[1]), dtype=np.float32)

        for row_idx in range(st_row, en_row + 1):
            for col_idx in range(data.shape[1]):
                sample_x, sample_y = Calc.__get_sample_phy_pos(calc, col_idx, row_idx)
                potential = Calc.__get_potential(calc, sample_x, sample_y)

                buf[row_idx - st_row][col_idx] += potential

            lock.acquire()
            n_done_row[0] += 1
            lock.release()

        lock.acquire()
        data[st_row:en_row + 1] += buf
        lock.release()


@cuda.jit('void(float32[:], float32[:,:], float32[:,:], int32[:])')
def gpu_kernel(phy_rect, data, charges, cnt):
    x, y = cuda.grid(2)
    if not (x < data.shape[1] and y < data.shape[0]):
        return

    n_row, n_col = data.shape

    sample_x = phy_rect[0] + (phy_rect[2] - phy_rect[0]) * x / (n_col - 1)
    sample_y = phy_rect[1] - (phy_rect[1] - phy_rect[3]) * y / (n_row - 1)

    res = 0.0
    for charge in charges:
        res += Charge.gpu_get_potential(sample_x, sample_y, charge)

    data[y][x] += res

    cuda.atomic.add(cnt, 0, 1)

    # n_data = data.shape[0] * data.shape[1]
    # if cnt[0] % 10000000 == 0 or cnt[0] == n_data:
    #     print('(', round(cnt[0] / n_data * 100, 3), '%)')
