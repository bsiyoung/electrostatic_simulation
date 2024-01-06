from __future__ import annotations
from typing import Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Charge import ChargeDist

import numpy as np
from PIL import Image

from Calc import Calc

import plots.potential_color as potential_color
import plots.potential_contour as potential_contour


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
        Adjust physical rect to quantized scale by mpp and down sampling value
        Init data array
        Init image array

        :return: None
        """

        # Adjust physical rect and get data shape
        sizes = Simulation.get_adjusted_size(self.phy_rect, self.down_sampling, self.mpp)
        self.phy_rect = sizes['adj_phy_rect']
        self.full_phy_rect = sizes['full_adj_phy_rect']
        n_data_row, n_data_col = sizes['data_shape']

        # Init data array
        self.data = np.zeros(sizes['data_shape'], dtype=np.float32)

        # Init image array
        n_img_row = n_data_row * self.down_sampling
        n_img_col = n_data_col * self.down_sampling
        self.img = np.zeros((n_img_row, n_img_col, 3), dtype=np.uint8)
        self.img.fill(255)

    @staticmethod
    def get_adjusted_size(phy_rect: Tuple[float, float, float, float], down_sampling: int, mpp: float) -> dict:
        """
        Get
        physical rect which adjusted to quantized scale by mpp and down sampling value
        and shape of data

        n_data_row, n_data_col : shape of data
        adj_phy_rect : boundary of sampling points in physical domain
        full_adj_phy_rect : boundary of full plot in physical domain
        => if down_sampling is 1, phy_rect and full_phy_rect will be same

        :return: dictionary of data shape and adjusted physical rects
        """

        # meter per data (1 data value : data of mpd x mpd size square area)
        mpd = down_sampling * mpp

        # Center point of physical rect (won't be changed)
        phy_cntr = (
            (phy_rect[0] + phy_rect[2]) / 2,
            (phy_rect[1] + phy_rect[3]) / 2
        )

        phy_width = abs((phy_rect[2] - phy_rect[0]) / 2)
        phy_height = abs((phy_rect[3] - phy_rect[1]) / 2)

        # Size of data array
        half_n_data_col = int(np.ceil((phy_width / 2) / mpd))
        half_n_data_row = int(np.ceil((phy_height / 2) / mpd))
        n_data_col = half_n_data_col * 2
        n_data_row = half_n_data_row * 2

        # Adjusted physical rect
        half_adj_phy_width = (half_n_data_col - 0.5) * mpd
        half_adj_phy_height = (half_n_data_row - 0.5) * mpd

        adj_phy_rect = (
            phy_cntr[0] - half_adj_phy_width,
            phy_cntr[1] + half_adj_phy_height,
            phy_cntr[0] + half_adj_phy_width,
            phy_cntr[1] - half_adj_phy_height
        )

        # Adjusted full physical rect
        phy_margin = (down_sampling / 2 - 0.5) * mpp
        full_adj_phy_rect = (
            adj_phy_rect[0] - phy_margin,
            adj_phy_rect[1] + phy_margin,
            adj_phy_rect[2] + phy_margin,
            adj_phy_rect[3] - phy_margin
        )

        return {
            'data_shape': (n_data_row, n_data_col),
            'image_size': (n_data_row * down_sampling, n_data_col * down_sampling),
            'adj_phy_rect': adj_phy_rect,
            'full_adj_phy_rect': full_adj_phy_rect
        }

    def run(self) -> None:
        # Fill data array
        self.calc.do()

        # Pile up plots
        plot_module = {
            'potential_color': potential_color,
            'potential_contour': potential_contour
        }
        for plot, conf in self.plots.items():
            if self.device == 'cpu':
                plot_module[plot].cpu(self.img, self.data, conf)
            elif self.device == 'gpu':
                plot_module[plot].gpu(self.img, self.data, conf)

        # Save image
        res = Image.fromarray(self.img)
        res.save('result.png')
