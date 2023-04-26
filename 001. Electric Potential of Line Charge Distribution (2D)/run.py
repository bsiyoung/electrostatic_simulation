import sys
import math

import plot

# Reference Point of The Electric Potential is at Infinity

# Set Plot Range (unit : m)
# x range : -plot_size ~ plot_size
# y range : -plot_size ~ plot_size
plot_size = 20
window_size = 900
grid_size = 0.05

# Define Surface Charge Distribution
p1 = (-4, -4)  # Line from p1 to p2
p2 = (4, 4)
surface_charge_density = 1e-10  # unit : C/m^2
z_depth = 1e-6  # unit : m

# Electric Permittivity (unit : F/m)
eps = 8.8541878128e-12

# Small Delta Value To Avoid Divide By Zero Error
delta = 1e-10


def run():
    potential_data = get_data()
    plot.draw_plot(potential_data, (p1, p2),
                   plot_size, window_size, grid_size,
                   'result.png')


def get_data():
    data = []

    n = int(plot_size // grid_size) + 2
    num_data = (2 * n) ** 2
    min_value = None
    max_value = None
    for rx_n in range(-n, n):
        for ry_n in range(-n, n):
            # Get Electric Potential at Center of Grid
            rx = rx_n * grid_size + grid_size / 2
            ry = ry_n * grid_size + grid_size / 2

            potential = get_potential((rx, ry))
            data.append([(rx_n, ry_n), potential])

            if min_value is None:
                min_value = max_value = potential
            elif potential < min_value:
                min_value = potential
            elif max_value < potential:
                max_value = potential

            curr_data_idx = (rx_n + n) * (2 * n) + (ry_n + n) + 1
            if curr_data_idx % 10 == 0 or curr_data_idx == num_data:
                perc = round(curr_data_idx / num_data * 100, 3)
                print('\r{}/{} ({}%)'.format(curr_data_idx, num_data, perc), end='')
    print()

    return {'min': min_value, 'max': max_value, 'data': data}


def get_potential(r):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    line_len = math.sqrt(dx ** 2 + dy ** 2)

    # Affine Transform
    _rx_tmp = r[0] - p1[0]
    _ry_tmp = r[1] - p1[1]

    theta = -math.atan2(dy, dx)
    _rx = _rx_tmp * math.cos(theta) - _ry_tmp * math.sin(theta)
    _ry = _rx_tmp * math.sin(theta) + _ry_tmp * math.cos(theta)

    # Calculate Pre-calculated Integral
    _z = z_depth / 2

    a0 = line_len - _rx
    a1 = -_rx

    b0 = math.sqrt(a0 ** 2 + _ry ** 2 + _z ** 2)
    b1 = math.sqrt(a1 ** 2 + _ry ** 2 + _z ** 2)

    c00 = a0 * math.log(b0 + _z)
    c01 = _ry * math.atan((_z * a0) / (_ry * b0 + delta))
    c02 = _z * math.atanh(a0 / (b0 + delta))

    c10 = a1 * math.log(b1 - _z)
    c11 = _ry * math.atan((-_z * a1) / (_ry * b1 + delta))
    c12 = -_z * math.atanh(a1 / (b1 + delta))

    res1 = c00 - c01 + c02
    res2 = c10 - c11 + c12

    res = surface_charge_density / (4 * math.pi * eps) * (res2 - res1)
    return res


if __name__ == '__main__':
    run()
