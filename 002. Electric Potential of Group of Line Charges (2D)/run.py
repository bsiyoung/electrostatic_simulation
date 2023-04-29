import json

import plot
import electrostatics

# Reference Point of The Electric Potential is at Infinity

# Set Plot Range (unit : m)
# x range : -plot_size ~ plot_size
# y range : -plot_size ~ plot_size
plot_size = 20
window_size = 900
grid_size = 0.05


def run():
    f = open('charges.json', 'r')
    json_data = json.load(f)
    f.close()

    potential_data = get_data(json_data)
    plot.draw_plot(potential_data, json_data,
                   plot_size, window_size, grid_size,
                   'result.png')


def get_data(line_info):
    n = int(plot_size // grid_size) + 2
    data = [[(i // (2 * n) - n, i % (2 * n) - n), 0] for i in range((2 * n) ** 2)]

    num_data = (2 * n) ** 2
    line_keys = list(line_info.keys())
    for no_line, line in enumerate(line_keys):
        for rx_n in range(-n, n):
            for ry_n in range(-n, n):
                # Position of Center of Current Grid
                rx = rx_n * grid_size + grid_size / 2
                ry = ry_n * grid_size + grid_size / 2

                # Accumulate The Potential
                potential = electrostatics.get_potential(line_info[line], (rx, ry))
                data_idx = (rx_n + n) * (2 * n) + (ry_n + n)
                data[data_idx][1] += potential

                # Print Progress
                if (data_idx + 1) % 10 == 0 or (data_idx + 1) == num_data:
                    perc = round((data_idx * (no_line + 1) + 1) / (num_data * len(line_keys)) * 100, 3)
                    print('\r{}/{} ({}%)'.format(data_idx + num_data * no_line + 1,
                                                 num_data * len(line_keys), perc), end='')
    print()

    # Get Potential Range
    min_value = None
    max_value = None
    for i in range(len(data)):
        if min_value is None:
            min_value = max_value = data[i][1]
        elif min_value > data[i][1]:
            min_value = data[i][1]
        elif max_value < data[i][1]:
            max_value = data[i][1]

    return {'min': min_value, 'max': max_value, 'data': data}


if __name__ == '__main__':
    run()
