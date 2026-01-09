class Module:
    def __init__(self):
        self.parameters = []

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def backward(self, x):
        raise NotImplementedError

    def parameters(self):
        return self.parameters