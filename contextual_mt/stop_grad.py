import torch


class StopGrad(torch.autograd.Function):
    """
    A layer that prevents gradient flow in a backward pass. Zeroes the gradient tensor.
    """

    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 0


class StopGradLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fn = StopGrad.apply

    def forward(self, x):
        return self.fn(x)
