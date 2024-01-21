from queue import Queue

from Simulation import Simulation
from Charge import ChargeDist


def run():
    sim_conf = {
        'phy_rect': (-0.4, 0.4, 0.4, -0.4),
        'mpp': 0.5e-4,
        'down_sampling': 1,
        'plots': {
            'potential_color': {
                 'min': (0, 0, 255),
                 'max': (255, 0, 0),
                 'ref': (255, 255, 255)
            },
            'potential_contour': {
                'scale': 0.5
            }
        },
        'ref_point': None,
        'device': 'cpu',
        'charges': [
            ChargeDist(-0.1, 0.05, 0.1, 0.05, density=1e-8, depth=0.4),
            ChargeDist(-0.1, -0.05, 0.1, -0.05, density=-1e-8, depth=0.4)
        ]
    }

    print('Image Size : ', Simulation.get_adjusted_size(sim_conf['phy_rect'],
                                                        sim_conf['down_sampling'],
                                                        sim_conf['mpp'])['image_size'])

    progress_q = Queue()
    sim = Simulation(sim_conf)
    sim.run(progress_q, verbose=True)


if __name__ == '__main__':
    run()
