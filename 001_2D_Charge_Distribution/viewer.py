import tkinter as tk
import pyglet

import numpy as np


class Viewer:
    def __init__(self, title, width, height):
        # Load font file
        if not pyglet.font.have_font('D2Coding'):
            pyglet.font.add_file('../fonts/D2Coding-Ver1.3.2-20180524.ttf')

        # Initialize
        self.wnd_w = width
        self.wnd_h = height

        self.__init_main_window(title)
        self.__init_canvas()
        self.__init_default_configs()

        # Pixel value info
        self.pixel_value = []

        # Initialization
        self.reset_pixel_values()
        self.redraw_canvas()

    def __init_default_configs(self):
        self.wnd_cntr_pos = [0, 0]  # physical position of window center
        self.mpp = 1e-3  # meter per pixel, 1px = 1mm
        self.min_mpp = 5e-7
        self.max_mpp = 1
        self.drag_prev_pos = [None, None]
        self.zoom_speed = 1.2  # scroll speed

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

    def __init_main_window(self, title):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry('{}x{}'.format(self.wnd_w, self.wnd_h))
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
        self.reset_pixel_values()

        self.redraw_canvas()

    # Canvas initialization & bindings
    # ==================================================================================================================
    def __init_canvas(self):
        self.canvas = tk.Canvas(self.root, width=self.wnd_w, height=self.wnd_h, bg='white')
        self.__canvas_bind_funcs()
        self.canvas.pack()

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

        self.redraw_canvas()

    def __left_mouse_down(self, event):
        self.drag_prev_pos = [event.x, event.y]

    def __left_mouse_drag(self, event):
        dx = event.x - self.drag_prev_pos[0]
        dy = event.y - self.drag_prev_pos[1]

        self.wnd_cntr_pos[0] -= dx * self.mpp
        self.wnd_cntr_pos[1] += dy * self.mpp

        self.drag_prev_pos = [event.x, event.y]

        self.redraw_canvas()

    def __left_mouse_up(self, event):
        self.drag_prev_pos = [None, None]

    def reset_pixel_values(self):
        self.pixel_value = np.empty((self.wnd_h, self.wnd_w), dtype=np.float32)

    def redraw_canvas(self):
        self.canvas.delete('all')

        grid_size = self.axis_conf['grid']['size']
        self.draw_axes(grid_size)

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
            self.canvas.create_line(0, x_axis_pos, self.wnd_w, x_axis_pos,
                                    fill=axis_conf['color'], width=axis_conf['width'])

        if is_y_axis_visible:
            # Draw y-axis
            self.canvas.create_line(y_axis_pos, 0, y_axis_pos, self.wnd_h,
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
        wnd_right_phy_x, wnd_bot_phy_y = self.screen_pos_to_physical_pos(self.wnd_w, self.wnd_h)

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

    def run_mainloop(self):
        self.root.mainloop()
