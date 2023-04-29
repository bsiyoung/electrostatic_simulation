import math

# Electric Permittivity (unit : F/m)
eps = 8.8541878128e-12

# Small Delta Value To Avoid Divide By Zero Error
delta = 1e-10

# Z axis Width of Line Charge To Avoid An Infinity Potential
z_depth = 1e-10  # unit : m


def get_potential(line_info, r):
    p1 = (line_info['from']['x'], line_info['from']['y'])
    p2 = (line_info['to']['x'], line_info['to']['y'])

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

    res = line_info['charge_density'] / (4 * math.pi * eps) * (res2 - res1)
    return res
