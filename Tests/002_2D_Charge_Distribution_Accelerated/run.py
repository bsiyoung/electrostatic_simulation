from Simulation import Simulation
from Charge import ChargeDist


def run():
    sim_conf = {
        'phy_rect': (-0.4, 0.4, 0.4, -0.4),
        'mpp': 1e-4,
        'down_sampling': 1,
        'plots': {
            'color': {
                'min': (0, 0, 255),
                'max': (255, 0, 0),
                'ref': (255, 255, 255)
            },
            'contour': {
                'scale': 3
            }
        },
        'ref_point': None,
        'device': 'gpu',
        'charges': [
            ChargeDist(-0.1, 0.05, 0.1, 0.05, density=1e-8, depth=0.2),
            ChargeDist(-0.1, -0.05, 0.1, -0.05, density=-1e-8, depth=0.2),
            ChargeDist(-0.2, -0.1, -0.2, 0.1, density=2e-8, depth=0.2),
            ChargeDist(0.2, -0.1, 0.2, 0.1, density=-2e-8, depth=0.2)
        ]
    }

    sim = Simulation(sim_conf)
    sim.run()


if __name__ == '__main__':
    run()
