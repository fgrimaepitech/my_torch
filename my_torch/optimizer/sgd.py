from typing import Callable
import numpy as np
from my_torch.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0, dampening: float = 0,  weight_decay: float = 0.0, nesterov: bool = False):
        default = {
            'lr': lr,
            'momentum': momentum,
            'weight_decay': weight_decay,
            'dampening': dampening,
            'nesterov': nesterov
        }
        super().__init__(params, default)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('dampening', 0)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is not None:
                    continue

                grad = p.grad

                if weight_decay != 0:
                    grad = grad + weight_decay * p.data

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = np.zeros_like(p.data)
                    else:
                        buf = param_state['momentum_buffer']
                    buf = momentum * buf + (1 - dampening) * grad
                    param_state['momentum_buffer'] = buf
                    if nesterov:
                        grad = grad + momentum * buf
                    else:
                        grad = buf
                p.data = p.data - lr * grad
        return loss

    
