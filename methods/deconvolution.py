from methods import baseoperations
from methods import backpropagation


class Deconvnet(backpropagation.BackPropagation):

    def __init__(self, model):
        super(Deconvnet, self).__init__(model)

        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, baseoperations.nn.ReLU):
                return (baseoperations.torch.clamp(grad_out[0], min=0.0),)

        for module in self.model.named_modules():
            self.handlers.append(module[1].register_backward_hook(backward_hook))

