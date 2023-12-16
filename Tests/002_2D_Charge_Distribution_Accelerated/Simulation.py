from __future__ import annotations
from typing import Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Charge import ChargeDist

import numpy as np
from numba import cuda
from PIL import Image

from Calc import Calc


class Simulation:
    def __init__(self, conf: dict):
        self.charges: Tuple[ChargeDist] = conf['charges']
        self.phy_rect: Tuple[float, float, float, float] = conf['phy_rect']
        self.full_phy_rect: Tuple[float, float, float, float] = (0, 0, 0, 0)
        self.mpp: float = conf['mpp']
        self.down_sampling: int = conf['down_sampling']
        self.plots: dict = conf['plots']
        self.device = conf['device']

        self.__init_data()
        self.calc: Calc = Calc(charges=self.charges,
                               phy_rect=self.phy_rect,
                               data=self.data,
                               ref_point=conf['ref_point'],
                               device=self.device)

    def __init_data(self) -> None:
        """
        Adjust physical rect and init data array and image

        :return: None
        """

        # Adjust physical rect and get data shape
        n_data_col, n_data_row = self.__adjust_physical_rect()

        # Init data array
        self.data = np.zeros((n_data_row, n_data_col), dtype=np.float32)

        # Init image
        n_img_row = n_data_row * self.down_sampling
        n_img_col = n_data_col * self.down_sampling
        self.img = np.zeros((n_img_row, n_img_col, 3), dtype=np.uint8)
        self.img.fill(255)

    def __adjust_physical_rect(self) -> Tuple[int, int]:
        """
        Adjust physical rect to fit mpp and down_sampling values

        phy_rect : boundary of sampling points in physical domain
        full_phy_rect : boundary of full plot in physical domain

        => if down_sampling is 1, phy_rect and full_phy_rect will be same

        :return: shape of data
        """

        # Center point of physical rect (won't change)
        self.phy_cntr: Tuple[float, float] = (
            (self.phy_rect[0] + self.phy_rect[2]) / 2,
            (self.phy_rect[1] + self.phy_rect[3]) / 2
        )

        # Size of data shape
        half_n_data_col = np.ceil((self.phy_cntr[0] - self.phy_rect[0]) / (self.down_sampling * self.mpp))
        half_n_data_row = np.ceil((self.phy_rect[1] - self.phy_cntr[1]) / (self.down_sampling * self.mpp))

        half_n_data_col = int(half_n_data_col)
        half_n_data_row = int(half_n_data_row)

        n_data_col = 2 * half_n_data_col
        n_data_row = 2 * half_n_data_row

        # Adjusted physical rect to fit mpp and down_sampling
        half_adj_phy_width = (half_n_data_col - 0.5) * self.down_sampling * self.mpp
        half_adj_phy_height = (half_n_data_row - 0.5) * self.down_sampling * self.mpp

        adj_phy_rect = (
            self.phy_cntr[0] - half_adj_phy_width,
            self.phy_cntr[1] + half_adj_phy_height,
            self.phy_cntr[0] + half_adj_phy_width,
            self.phy_cntr[1] - half_adj_phy_height
        )
        self.phy_rect = adj_phy_rect

        # Get full adjusted physical rect
        phy_margin = (self.down_sampling / 2 - 0.5) * self.mpp
        self.full_phy_rect = (
            self.phy_rect[0] - phy_margin,
            self.phy_rect[1] + phy_margin,
            self.phy_rect[2] + phy_margin,
            self.phy_rect[3] - phy_margin
        )

        return n_data_col, n_data_row

    def run(self):
        self.calc.do()

        plot_func_pair = {
            'color': {'cpu': color_plot, 'gpu': gpu_color_plot},
            'contour': {'cpu': contour_plot, 'gpu': gpu_contour_plot}
        }
        for plot, conf in self.plots.items():
            plot_func_pair[plot][self.device](self.img, self.data, conf)

        res = Image.fromarray(self.img)
        res.save('result.png')


def color_plot(img, data, conf):
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
    for row_idx in range(data_shape[0]):
        r0 = row_idx * down_sampling
        r1 = r0 + down_sampling
        for col_idx in range(data_shape[1]):
            c0 = col_idx * down_sampling
            c1 = c0 + down_sampling
            img[r0:r1, c0:c1] = get_color(data[row_idx][col_idx])


def gpu_color_plot(img, data, conf):
    min_value = np.min(data)
    max_value = np.max(data)
    max_abs = max(abs(min_value), abs(max_value))

    data_shape = data.shape
    down_sampling = img.shape[0] // data_shape[0]

    d_img = cuda.to_device(img)
    d_data = cuda.to_device(data)

    d_min_color = cuda.to_device(np.array(conf['min'], dtype=np.int32))
    d_ref_color = cuda.to_device(np.array(conf['ref'], dtype=np.int32))
    d_max_color = cuda.to_device(np.array(conf['max'], dtype=np.int32))

    n_thread_in_block = (16, 16)
    n_block_in_grid = (
        data.shape[1] // n_thread_in_block[0] + 1,
        data.shape[0] // n_thread_in_block[1] + 1
    )
    gpu_color_plot_kernel[n_block_in_grid, n_thread_in_block](d_img, d_data, down_sampling, max_abs,
                                                              d_min_color, d_ref_color, d_max_color)
    cuda.synchronize()

    img[:] = d_img.copy_to_host()


@cuda.jit('void(uint8[:,:,:], float32[:,:], int32, float32, int32[:], int32[:], int32[:])')
def gpu_color_plot_kernel(img, data, down_sampling, max_abs, min_color, ref_color, max_color):
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


def contour_plot(img, data, conf):
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


def gpu_contour_plot(img, data, conf):
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
    gpu_contour_plot_kernel[n_block_in_grid, n_thread_in_block](d_img, d_data, scale, down_sampling)
    cuda.synchronize()

    img[:] = d_img.copy_to_host()


@cuda.jit('void(uint8[:,:,:], float32[:,:], int32, int32)')
def gpu_contour_plot_kernel(img, data, scale, down_sampling):
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
