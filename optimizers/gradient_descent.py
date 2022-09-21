import torch
from torch.optim.optimizer import Optimizer
from torch import autograd

import optimizers.utilities as utils

class GradientDescent(Optimizer):

    def __init__(self, params, lr=1e-2, minibatch=None, momentum=0.0, nesterov=False,
                 weight_decay=0.0, **kwargs):
        defaults = dict(lr=lr, minibatch=minibatch, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay, **kwargs)
        super(GradientDescent, self).__init__(params, defaults)

    def step(self, fn, data):
        if len(self.param_groups) != 1:
            raise ValueError("Only one parameter group is currently supported.")
        group = self.param_groups[0]

        self.state["gradient"], (outputs, loss) = self._grad(fn, data, group["params"])
        self._step()

        return outputs, loss

    def _grad(self, fn, data, params, weight_decay=True):
        group = self.param_groups[0]

        grad_ = lambda outputs, loss: autograd.grad(loss, params)
        g, (outputs, loss) = utils.minibatch(grad_, fn, data, batch=group["minibatch"])

        if weight_decay and group["weight_decay"] != 0.0:
            torch._foreach_add_(g, group["params"], alpha=group["weight_decay"])
        return g, (outputs, loss)

    @torch.no_grad()
    def _step(self):
        group = self.param_groups[0]
        params = group["params"]

        if group["momentum"] == 0.0:
            torch._foreach_add_(self.state["gradient"], alpha=-group["lr"])

        else:
            if "momentum" not in self.state:  # initialize momentum variable
                self.state["momentum"] = utils.identity(params, fill=0.0)
            # momentum update
            torch._foreach_mul_(self.state["momentum"], group["momentum"])
            torch._foreach_add_(self.state["momentum"], self.state["gradient"], alpha=group["lr"])
            # parameters update
            if group["nesterov"]:
                update = torch._foreach_mul(self.state["momentum"], group["momentum"])
                torch._foreach_add_(update, self.state["gradient"], alpha=group["lr"])
                torch._foreach_sub_(params, update)
                del update
            else:
                torch._foreach_sub_(params, self.state["momentum"])