from ast import Dict, List
from typing import Any, Callable

class Optimizer:
    def __init__(self, params, defaults: Dict[str, Any]):
        self.defaults = defaults
        self.param_groups = []
        self.state = {}
        self._hook_pre_hooks: List[Callable] = []
        self._hook_post_hooks: List[Callable] = []

        if isinstance(params, (list, tuple)):
            for param in params:
                self.add_param_group({'params': param})
        else:
            self.add_param_group({'params': params})

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