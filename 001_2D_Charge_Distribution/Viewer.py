from __future__ import annotations
from typing import TYPE_CHECKING
from typing import List

import time

import tkinter as tk
import pyglet
import numpy as np
from PIL import Image, ImageTk

from CalcManager import CalcJob

if TYPE_CHECKING:
    from DrawManager import DrawManager, DrawRect
    from CalcManager import CalcManager


class Viewer:
    def __init__(self, title: str, width: int, height: int,
                 draw_mgr: DrawManager, calc_mgr: CalcManager,
                 exit_flag: List[bool]):
        self.exit_flag: List[bool] = exit_flag
        self.draw_mgr: DrawManager = draw_mgr
        self.calc_mgr: CalcManager = calc_mgr

        self.wnd_w: int = width
        self.wnd_h: int = height

        self.root: tk.Tk = tk.Tk()
        self.__init_main_window(title)

        self.canvas: tk.Canvas = tk.Canvas(self.root, width=width, height=height, bg='white')
        self.data: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        self.data_img_obj: ImageTk.PhotoImage = ImageTk.PhotoImage(image=Image.fromarray(self.data))
        self.data_img_id: int | None = None
        self.__init_canvas()

        self.__init_default_configs()

        self.__request_update(0, 0, width - 1, height - 1)

    def __init_default_configs(self):
        self.wnd_cntr_pos = [0, 0]  # physical position of window center
        self.mpp = 1e-3  # meter per pixel, 1px = 1mm
        self.min_mpp = 5e-7
        self.max_mpp = 1
        self.drag_prev_pos = [None, None]
        self.zoom_speed = 1.2  # scroll speed

        self.max_refresh_rate = int(1000 / 60)  # ms

        self.axis_conf = {
            'main': {
                'width': 1.5,
                'color': '#000000'
            },
            'grid': {
                'font': {
                    'family': 'D2Coding',
                    'size': 10,
                    'color': '#404040'
                },
                'size': self.wnd_w // 6,  # in pixel
                'width': 1,
                'color': '#B0B0B0',
                'decimal': 2,
                'decimal_conf': [
                    [1e-0, 2],
                    [1e-1, 3],
                    [1e-2, 4],
                    [1e-3, 5],
                    [1e-4, 6],
                    [1e-5, 7],
                    [1e-6, 8]
                ],
                'text_margin': 5
            }
        }

    def __request_update(self, x1, y1, x2, y2):
        self.calc_mgr.calc_queue.clear()
        self.calc_mgr.max_abs = 1e-100

        self.draw_mgr.lock.acquire()
        self.draw_mgr.draw_queue.clear()
        self.draw_mgr.lock.release()

        px1, py1 = self.screen_pos_to_physical_pos(x1, y1)
        px2, py2 = self.screen_pos_to_physical_pos(x2, y2)
        self.calc_mgr.calc_queue.append((
            (
                CalcJob((x1, y1, x2, y2), (px1, py1, px2, py2),0),
            ),
            None
        ))

    def __clear_data(self):
        self.data.fill(255)

    # Window initialization & binding
    # ==================================================================================================================
    def __init_main_window(self, title):
        self.root.title(title)
        self.root.geometry('{}x{}'.format(self.wnd_w, self.wnd_h))
        self.root.protocol('WM_DELETE_WINDOW', self.__on_destroy)
        self.root.bind('<Configure>', self.__window_resize)

    def __window_resize(self, event):
        new_width = event.width
        new_height = event.height

        if new_width == self.wnd_w and new_height == self.wnd_h:
            # Return if there is no size change
            return

        self.wnd_w = new_width
        self.wnd_h = new_height
        self.canvas.configure(width=new_width, height=new_height)

        self.__update_data_size()

    def __on_destroy(self):
        self.exit_flag[0] = True
        self.root.destroy()

    # Canvas initialization & binding
    # ==================================================================================================================
    def __init_canvas(self):
        # Load font
        if not pyglet.font.have_font('D2Coding'):
            pyglet.font.add_file('../fonts/D2Coding-Ver1.3.2-20180524.ttf')

        self.__init_data_image()
        self.__canvas_bind_funcs()
        self.canvas.pack()

    def __update_data_size(self):
        shape = self.data.shape

        if shape[0] == self.wnd_h and shape[1] == self.wnd_w:
            return

        new_data = np.zeros((self.wnd_h, self.wnd_w, 3), dtype=np.uint8)
        new_data.fill(255)

        if self.wnd_w > shape[1]:
            src_st_x = 0
            dst_st_x = self.wnd_w // 2 - shape[1] // 2
            copy_width = shape[1]
        else:
            src_st_x = shape[1] // 2 - self.wnd_w // 2
            dst_st_x = 0
            copy_width = self.wnd_w

        if self.wnd_h > shape[0]:
            src_st_y = 0
            dst_st_y = self.wnd_h // 2 - shape[0] // 2
            copy_height = shape[0]
        else:
            src_st_y = shape[0] // 2 - self.wnd_h // 2
            dst_st_y = 0
            copy_height = self.wnd_h

        new_data[dst_st_y:dst_st_y + copy_height, dst_st_x:dst_st_x + copy_width] \
            = self.data[src_st_y:src_st_y + copy_height, src_st_x:src_st_x + copy_width]

        pos_dx = self.wnd_w // 2 - shape[1] // 2
        pos_dy = self.wnd_h // 2 - shape[0] // 2
        self.canvas.move(self.data_img_id, pos_dx, pos_dy)
        self.data = new_data

    def __init_data_image(self):
        self.__clear_data()
        self.data_img_obj = ImageTk.PhotoImage(image=Image.fromarray(self.data))
        self.data_img_id = self.canvas.create_image(0, 0, image=self.data_img_obj, anchor=tk.NW)

    def __canvas_bind_funcs(self):
        self.canvas.bind('<MouseWheel>', self.__mouse_scroll)
        self.canvas.bind('<Button-1>', self.__left_mouse_down)
        self.canvas.bind('<B1-Motion>', self.__left_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.__left_mouse_up)

    def __mouse_scroll(self, event):
        """
        Perform zoom in/out when mouse scrolled

        :param event: Mouse scroll event object
        :return: None
        """

        zoom_ratio = abs(event.delta / 120 * self.zoom_speed)
        is_zoom_in = event.delta > 0

        if is_zoom_in is True:
            self.mpp *= zoom_ratio
        else:
            self.mpp /= zoom_ratio

        self.mpp = np.clip(self.mpp, self.min_mpp, self.max_mpp)

        # Dynamically adjust num of decimal by mpp value
        decimal_setting = self.axis_conf['grid']['decimal_conf']
        decimal_setting.sort(key=lambda x: x[0], reverse=True)
        for setting in decimal_setting:
            if self.mpp <= setting[0]:
                self.axis_conf['grid']['decimal'] = setting[1]

        self.__clear_data()
        self.__request_update(0, 0, self.wnd_w - 1, self.wnd_h - 1)

    def __left_mouse_down(self, event):
        self.drag_prev_pos = [event.x, event.y]

    def __left_mouse_drag(self, event):
        dx = event.x - self.drag_prev_pos[0]
        dy = event.y - self.drag_prev_pos[1]

        self.wnd_cntr_pos[0] -= dx * self.mpp
        self.wnd_cntr_pos[1] += dy * self.mpp
        self.drag_prev_pos = [event.x, event.y]

        src_st_x = 0 if dx > 0 else -dx
        dst_st_x = dx if dx > 0 else 0
        copy_width = self.wnd_w - abs(dx)

        src_st_y = 0 if dy > 0 else -dy
        dst_st_y = dy if dy > 0 else 0
        copy_height = self.wnd_h - abs(dy)

        reset_region = [
            # Horizontal
            {
                'st_x': dx if dx > 0 else 0,
                'st_y': 0 if dy > 0 else self.wnd_h + dy,
                'width': copy_width,
                'height': abs(dy)
            },
            # Vertical
            {
                'st_x': 0 if dx > 0 else self.wnd_w + dx,
                'st_y': dy if dy > 0 else 0,
                'width': abs(dx),
                'height': copy_height
            },
            # Intersection
            {
                'st_x': 0 if dx > 0 else self.wnd_w + dx,
                'st_y': 0 if dy > 0 else self.wnd_h + dy,
                'width': abs(dx),
                'height': abs(dy)
            }
        ]

        self.data[dst_st_y:dst_st_y + copy_height, dst_st_x:dst_st_x + copy_width] \
            = self.data[src_st_y:src_st_y + copy_height, src_st_x:src_st_x + copy_width]

        for rect in reset_region:
            self.data[rect['st_y']:rect['st_y'] + rect['height'], rect['st_x']:rect['st_x'] + rect['width'], :] = 255

    def __left_mouse_up(self, event):
        self.drag_prev_pos = [None, None]

    # Draw axes
    # ==================================================================================================================
    '''
    draw_axes()
     ├─ __draw_main_axes()
     └─ __draw_grids()
         ├─ __get_grid_boundaries()    # Get info to draw grids
         ├─ __get_grid_text_sizes()
         ├─ __draw_vertical_grids()    # Draw grids
         └─ __draw_horizontal_grids()         
    '''

    def draw_axes(self, sz_grid):
        # Y position of X-axis in pixel
        y_axis_pos, x_axis_pos = self.physical_pos_to_screen_pos(0, 0, get_int=True)

        # Draw
        self.__draw_main_axes(x_axis_pos, y_axis_pos)
        self.__draw_grids(sz_grid, x_axis_pos, y_axis_pos)

    def __draw_main_axes(self, x_axis_pos, y_axis_pos):
        is_x_axis_visible = 0 <= x_axis_pos <= self.wnd_h
        is_y_axis_visible = 0 <= y_axis_pos <= self.wnd_w

        # Draw axes
        axis_conf = self.axis_conf['main']
        if is_x_axis_visible:
            # Draw x-axis
            self.canvas.create_line(0, x_axis_pos, self.wnd_w - 1, x_axis_pos,
                                    fill=axis_conf['color'], width=axis_conf['width'])

        if is_y_axis_visible:
            # Draw y-axis
            self.canvas.create_line(y_axis_pos, 0, y_axis_pos, self.wnd_h - 1,
                                    fill=axis_conf['color'], width=axis_conf['width'])

    def __draw_grids(self, sz_grid, x_axis_pos, y_axis_pos):
        sz_grid_px = sz_grid  # Grid size in pixel
        sz_grid_phy = sz_grid_px * self.mpp  # Grid size in meter

        # Get draw boundaries of grid
        st_wnd_x, en_wnd_x, st_wnd_y, en_wnd_y = self.__get_grid_boundaries(sz_grid_phy)

        # Get sizes of grid text
        grid_text_h, grid_char_width = self.__get_grid_text_sizes()

        # Draw grid
        self.__draw_vertical_grids(sz_grid_px, st_wnd_x, en_wnd_x, y_axis_pos, grid_text_h)
        self.__draw_horizontal_grids(sz_grid_px, st_wnd_y, en_wnd_y, x_axis_pos, grid_char_width)

    def __get_grid_boundaries(self, sz_grid_phy):
        # Get physical position of edge of window
        wnd_left_phy_x, wnd_top_phy_y = self.screen_pos_to_physical_pos(0, 0)
        wnd_right_phy_x, wnd_bot_phy_y = self.screen_pos_to_physical_pos(self.wnd_w - 1, self.wnd_h - 1)

        # Get start and end locations of grid
        st_phy_x = wnd_left_phy_x - wnd_left_phy_x % sz_grid_phy
        en_phy_x = wnd_right_phy_x
        st_phy_y = wnd_top_phy_y - wnd_top_phy_y % sz_grid_phy
        en_phy_y = wnd_bot_phy_y

        st_wnd_x, st_wnd_y = self.physical_pos_to_screen_pos(st_phy_x, st_phy_y, get_int=True)
        en_wnd_x, en_wnd_y = self.physical_pos_to_screen_pos(en_phy_x, en_phy_y, get_int=True)

        return st_wnd_x, en_wnd_x, st_wnd_y, en_wnd_y

    def __get_grid_text_sizes(self):
        grid_conf = self.axis_conf['grid']
        grid_text_font = (grid_conf['font']['family'], grid_conf['font']['size'])

        tmp_canvas = tk.Canvas()
        test_text = '1234567890eE-'
        grid_text_obj = tmp_canvas.create_text(0, 0, text=test_text, font=grid_text_font)
        grid_text_rect = tmp_canvas.bbox(grid_text_obj)
        grid_text_h = grid_text_rect[3] - grid_text_rect[1]
        grid_char_width = (grid_text_rect[2] - grid_text_rect[0]) // len(test_text)
        del tmp_canvas

        return grid_text_h, grid_char_width

    def __draw_vertical_grids(self, sz_grid_px, st_wnd_x, en_wnd_x, y_axis_pos, grid_text_h):
        grid_conf = self.axis_conf['grid']

        v_grid_text_pos = grid_text_h // 2 + grid_conf['text_margin']
        grid_text_font = (grid_conf['font']['family'], grid_conf['font']['size'])

        for x in range(st_wnd_x, en_wnd_x + 1, sz_grid_px):
            if abs(x - y_axis_pos) < sz_grid_px - 1:  # Don't draw y-axis
                continue

            self.canvas.create_line(x, 0, x, self.wnd_h,
                                    fill=grid_conf['color'], width=grid_conf['width'], dash=(6, 4))

            grid_phys_x = self.screen_x_to_physical_x(x)
            phys_x_str = ('{:.' + str(grid_conf['decimal']) + 'e}').format(grid_phys_x)
            self.canvas.create_text(x, v_grid_text_pos, text=phys_x_str,
                                    fill=grid_conf['font']['color'], font=grid_text_font)

    def __draw_horizontal_grids(self, sz_grid_px, st_wnd_y, en_wnd_y, x_axis_pos, grid_char_width):
        grid_conf = self.axis_conf['grid']

        grid_text_font = (grid_conf['font']['family'], grid_conf['font']['size'])

        for y in range(st_wnd_y, en_wnd_y + 1, sz_grid_px):
            if abs(y - x_axis_pos) < sz_grid_px - 1:  # Don't draw x-axis
                continue

            self.canvas.create_line(0, y, self.wnd_w, y,
                                    fill=grid_conf['color'], width=grid_conf['width'], dash=(6, 4))

            grid_phys_y = self.screen_y_to_physical_y(y)
            phys_y_str = ('{:.' + str(grid_conf['decimal']) + 'e}').format(grid_phys_y)
            h_grid_text_pos = grid_char_width * len(phys_y_str) // 2 + grid_conf['text_margin']
            self.canvas.create_text(h_grid_text_pos, y + grid_conf['text_margin'], text=phys_y_str,
                                    fill=grid_conf['font']['color'], font=grid_text_font)

    # Convert between physical and screen position
    # ==================================================================================================================
    '''
    physical_pos_to_screen_pos()
     ├─ physical_x_to_screen_x()
     └─ physical_y_to_screen_y()     
    '''

    def physical_pos_to_screen_pos(self, phy_x, phy_y, get_int=False):
        scr_x = self.physical_x_to_screen_x(phy_x, get_int=get_int)
        scr_y = self.physical_y_to_screen_y(phy_y, get_int=get_int)

        return scr_x, scr_y

    def physical_x_to_screen_x(self, phy_x, get_int=False):
        phy_x_diff = phy_x - self.wnd_cntr_pos[0]
        px_x_diff = phy_x_diff / self.mpp
        px_wnd_cntr_x = self.wnd_w / 2
        scr_x = px_wnd_cntr_x + px_x_diff

        if get_int is True:
            scr_x = int(scr_x)

        return scr_x

    def physical_y_to_screen_y(self, phy_y, get_int=False):
        phy_y_diff = phy_y - self.wnd_cntr_pos[1]
        px_y_diff = phy_y_diff / self.mpp
        px_wnd_cntr_y = self.wnd_h / 2
        scr_y = px_wnd_cntr_y - px_y_diff

        if get_int is True:
            scr_y = int(scr_y)

        return scr_y

    '''
    screen_pos_to_physical_pos()
     ├─ screen_x_to_physical_x()
     └─ screen_y_to_physical_y()     
    '''

    def screen_pos_to_physical_pos(self, scr_x, scr_y):
        phy_x = self.screen_x_to_physical_x(scr_x)
        phy_y = self.screen_y_to_physical_y(scr_y)

        return phy_x, phy_y

    def screen_x_to_physical_x(self, scr_x):
        px_x_diff = scr_x - self.wnd_w / 2
        phy_x_diff = px_x_diff * self.mpp
        phy_x = self.wnd_cntr_pos[0] + phy_x_diff

        return phy_x

    def screen_y_to_physical_y(self, scr_y):
        px_y_diff = scr_y - self.wnd_h / 2
        phy_y_diff = px_y_diff * self.mpp
        phy_y = self.wnd_cntr_pos[1] - phy_y_diff

        return phy_y

    # ==================================================================================================================
    def draw_loop(self):
        self.__update_data()
        self.__update_viewport()

        self.root.after(self.max_refresh_rate, self.draw_loop)

    def __update_data(self):
        self.draw_mgr.lock.acquire()
        draw_queue: List[DrawRect] = self.draw_mgr.draw_queue

        # Fill pixel data
        idx = 0
        while idx < len(draw_queue):
            r = draw_queue[idx]

            x1, y1 = r.scr_rect[0], r.scr_rect[1]
            x2, y2 = r.scr_rect[2], r.scr_rect[3]
            self.data[y1:y2 + 1, x1:x2 + 1] = r.color

            idx += 1

        # Delete already processed queue
        del draw_queue[:idx]
        self.draw_mgr.lock.release()

    def __update_viewport(self):
        self.data_img_obj = ImageTk.PhotoImage(image=Image.fromarray(self.data))
        self.canvas.itemconfig(self.data_img_id, image=self.data_img_obj)

        self.__update_axis()

    def __update_axis(self):
        old_axis_obj = list(self.canvas.find_all())
        old_axis_obj.remove(self.data_img_id)
        grid_size = self.axis_conf['grid']['size']
        self.draw_axes(grid_size)
        self.__remove_objects(old_axis_obj)

    def __remove_objects(self, obj_id_list):
        for obj_id in obj_id_list:
            self.canvas.delete(obj_id)

    def get_pixel_boundary(self):
        return 0, 0, self.wnd_w - 1, self.wnd_h - 1

    def get_physical_boundary(self):
        left, top = self.screen_pos_to_physical_pos(0, 0)
        right, bottom = self.screen_pos_to_physical_pos(self.wnd_w - 1, self.wnd_h - 1)

        return left, top, right, bottom

    def run_gui(self):
        self.root.after(0, self.draw_loop)
        self.root.mainloop()
