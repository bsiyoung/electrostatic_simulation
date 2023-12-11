from viewer import Viewer


def run():
    v = Viewer('test', 1000, 800)
    v.run_mainloop()


if __name__ == '__main__':
    run()
