
from ast import Dict, List
from typing import Callable, Any


class Module:
    def __init__(self):
        # Avoid naming collision with the parameters() method
        self._parameters = {}
        self._modules = {}

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def backward(self, x):
        raise NotImplementedError

    def train(self, mode: bool = True):
        self.training = mode
        for module in self._parameters:
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)
    
    def children(self):
        for name, module in self._modules.items():
            yield module
    
    def modules(self):
        yield self
        for child in self.children():
            yield from child.modules()

    def add_parameter(self, name, param):
        self._parameters[name] = param
    
    def add_module(self, name, module):
        self._modules[name] = module

    def parameters(self):
        params = []
        for param in self._parameters.values():
            if param.requires_grad:
                params.append(param)

        for module in self._modules.values():
            params.extend(module.parameters())

        return params

class Optimizer:
    def __init__(self, params, defaults: Dict[str, Any]):
        self.defaults = defaults
        self.param_groups = []
        self.state = {}
        self._hook_pre_hooks: List[Callable] = []
        self._hook_post_hooks: List[Callable] = []

        if isinstance(params, (list, tuple)) and len(params) > 0:
            if isinstance(params[0], dict) and 'params' in params[0]:
                for param_group in params:
                    merged_group = {**defaults, **param_group}
                    self.add_param_group(merged_group)
            else:
                for param in params:
                    self.add_param_group({**defaults, 'params': param})
        else:
            if isinstance(params, (list, tuple)) and len(params) == 0:
                self.add_param_group({**defaults, 'params': []})
            else:
                self.add_param_group({**defaults, 'params': params})

    def add_param_group(self, param_group: dict):
        self.param_groups.append(param_group)

    def load_state_dict(self, state_dict: dict):
        self.param_groups = state_dict['param_groups']
        self.state = state_dict['state']

    def register_load_state_dict_post_hook(self, hook: Callable, preprend: bool = False):
        if preprend:
            self.load_state_dict_post_hooks.insert(0, hook)
        else:
            self.load_state_dict_post_hooks.append(hook)

    def register_load_state_dict_pre_hook(self, hook: Callable, preprend: bool = False):
        if preprend:
            self.load_state_dict_pre_hooks.insert(0, hook)
        else:
            self.load_state_dict_pre_hooks.append(hook)

    def register_state_dict_post_hook(self, hook: Callable, preprend: bool = False):
        if preprend:
            self.state_dict_post_hooks.insert(0, hook)
        else:
            self.state_dict_post_hooks.append(hook)

    def register_state_dict_pre_hook(self, hook: Callable, preprend: bool = False):
        if preprend:
            self.state_dict_pre_hooks.insert(0, hook)
        else:
            self.state_dict_pre_hooks.append(hook)

    def register_step_post_hook(self, hook: Callable):
        self.step_post_hooks.append(hook)

    def register_step_pre_hook(self, hook: Callable):
        self.step_pre_hooks.append(hook)

    def state_dict(self) -> dict:
        return {
            'state': self.state,
            'param_groups': self.param_groups
        }

    def step(self, closure: Callable = None):
        if closure is not None:
            closure()
        else:
            for param_group in self.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        param.data = param.data - param.grad * param_group['lr']

    def zero_grad(self):
        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    param.grad.data.zero_()