
from ast import Dict, List
from typing import Callable, Any
from .device import get_device


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
    
    def to(self, device: str):
        """
        Move all parameters and buffers to the specified device.
        
        Args:
            device: 'cpu' or 'cuda'
        
        Returns:
            self (for method chaining)
        """
        device = get_device(device)
        
        # Move all parameters
        for name, param in self._parameters.items():
            if param is not None:
                self._parameters[name] = param.to(device)
        
        # Move all submodules recursively
        for name, module in self._modules.items():
            if module is not None:
                module.to(device)
        
        return self
    
    def cuda(self):
        """Move all parameters and buffers to GPU."""
        return self.to('cuda')
    
    def cpu(self):
        """Move all parameters and buffers to CPU."""
        return self.to('cpu')
    
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

    def state_dict(self):
        state = {}

        def collect_params(module, prefix=""):
            for name, param in module._parameters.items():
                if param is not None:
                    key = f"{prefix}{name}"
                    state[key] = param.data.copy()
            for name, submodule in module._modules.items():
                collect_params(submodule, f"{prefix}{name}.")

        collect_params(self)
        return state

    def load_state_dict(self, state_dict):
        def load_params(module, prefix=""):
            for name, param in module._parameters.items():
                if param is not None:
                    key = f"{prefix}{name}"
                    if key in state_dict:
                        param.data = state_dict[key].copy()
            
            for name, submodule in module._modules.items():
                load_params(submodule, f"{prefix}{name}.")
        
        load_params(self)

    def save(self, filepath):
        import numpy as np
        state_dict = self.state_dict()
        np.savez(filepath, **state_dict)

    @classmethod
    def load(cls, filepath):
        import numpy as np
        model = cls()
        data = np.load(filepath, allow_pickle=True)
        state_dict = {key: data[key] for key in data.keys()}
        
        model.load_state_dict(state_dict)
        print(f"Model loaded from {filepath}")
        
        return model

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
        from .device import get_array_module
        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    xp = get_array_module(param.device)
                    param.grad.data = xp.zeros_like(param.grad.data)