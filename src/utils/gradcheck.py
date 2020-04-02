class GradientCheck:
    def __init__(self, restart_threshold):
        self.restart_threshold = restart_threshold
        self.reset()

    def reset(self):
        self.grad_mean = 0
        self.grad_steps = 0

    def __call__(self, grad_norm):
        self.grad_steps += 1
        if self.grad_steps > 500 and grad_norm > self.restart_threshold * self.grad_mean:
            return True
        if self.grad_steps == 0:
            self.grad_mean = grad_norm
        else:
            self.grad_mean = 0.5 * self.grad_mean + 0.5 * grad_norm
        return False