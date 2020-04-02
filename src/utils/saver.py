import os
import torch
import numpy as np

class ModelSaver:
    def __init__(self, path, max_models = 5, sign = -1, comment = ""):
        self.path = path
        self.max_models = max_models
        self.history = []
        self.models = []
        self.model_idx = 0
        self.last_model = None
        self.best_model = None
        self.sign = sign

        with open(path + "/params.txt", "w") as f: f.write(comment)

    def save(self, model, value):
        if self.last_model != None:
            if len(self.history) == 0 or self.sign * value > max(self.history):
                if self.best_model != None:
                    os.remove(self.best_model)
                self.best_model = '%s/best%.5f.pt' % (self.path, value)
                torch.save(model, self.best_model)
            os.remove(self.last_model)

        self.last_model = '%s/epoch_%d-%.5f.pt' % (self.path, self.model_idx, value)
        torch.save(model, self.last_model)
        np.save(self.path + '/loss', np.array(self.history))
        
        self.history.append(self.sign * value)
        self.model_idx += 1