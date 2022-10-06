class Function(object):

    def __init__(self):

        # To be defined in children classes
        self.dimension = None
        self.mu = None
        self.L = None
        self.argmin = None
        self.min_value = None

    def forward(self, x):
        # Compute the value of the function on x
        raise NotImplementedError

    def backward(self, x):
        # Compute the gradient of the function on x
        raise NotImplementedError
