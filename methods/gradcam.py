from methods import baseoperations


class GradCAM(baseoperations.BaseOperation):

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = baseoperations.OrderedDict()
        self.grad_pool = baseoperations.OrderedDict()
        self.candidate_layers = candidate_layers 

        def forward_hook(key):
            def forward_hook_(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook_

        def backward_hook(key):  
            def backward_hook_(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook_

        for name, module in self.model.named_modules():
            if self.candidate_layers is None or name in self.candidate_layers:
                self.handlers.append(module.register_forward_hook(forward_hook(name)))
                self.handlers.append(module.register_backward_hook(backward_hook(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    def _compute_grad_weights(self, grads):
        return baseoperations.F.adaptive_avg_pool2d(grads, 1)

    def forward(self, image):
        self.image_shape = image.shape[2:]
        return super(GradCAM, self).forward(image)

    def generate(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = self._compute_grad_weights(grads)

        gcam = baseoperations.torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
        gcam = baseoperations.F.relu(gcam)

        gcam = baseoperations.F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


