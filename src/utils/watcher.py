import numpy as np
from PIL import Image
from visdom import Visdom

class Watcher:
    def __init__(self, var_dict):
        self.var_dict = var_dict
        self.viz = Visdom()

        self.scalars = []
        self.images = []

    def add_scalar(self, name):
        self.viz.line([0],[0],name,opts=dict(title=name))
        self.scalars.append(name)

    def add_image(self, name):
        self.viz.image(self.var_dict[name], win = name)
        self.images.append(name)

    def update(self, step, var_dict = None):
        if var_dict is not None: self.var_dict = var_dict

        for key in self.images:
            if key in self.var_dict:
                self.viz.image(
                    self.var_dict[key],
                    win=key
                )
        for key in self.scalars:
            if key in self.var_dict:
                self.viz.line(
                    [self.var_dict[key].item()],
                    [step],
                    win = key,
                    update='append'
                )



if __name__ == "__main__":
    class X:
        def __init__(self):
            self.A = 0
        def update(self):
            self.A -= 1
    x = X()

    watcher = Watcher(x.__dict__)
    watcher.add_scalar("A")

    for i in range(10):
        x.update()
        watcher.update(i+1)

