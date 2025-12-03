from copy import deepcopy

from tta.base import TTAMethod
import tta.submission as tent

import torch
from torch import nn


class NormTentThreePass(TTAMethod):

    def __init__(
        self,
        model,
        norm_eps=1e-5,
        norm_momentum=0.1,
        optim_steps=1,
        optim_lr=1e-3,
        optim_method="Adam",
        optim_beta=0.9,
        optim_momentum=0.9,
        optim_dampening=0.0,
        optim_nesterov=True,
        optim_wd=0.0,
        norm_batch_size=512,
        tent_batch_size=512,
    ):
        super().__init__(model)

        # norm configuration
        self.norm_eps = norm_eps
        self.norm_momentum = norm_momentum

        # tent configuration
        tent_params, _ = tent.collect_params(model)
        tent_optimizer = tent.setup_optimizer(
            tent_params,
            optim_lr,
            optim_method,
            optim_beta,
            optim_momentum,
            optim_dampening,
            optim_nesterov,
            optim_wd,
        )
        self.tent_optimizer = tent_optimizer
        self.tent_steps = optim_steps
        assert optim_steps > 0, "tent requires >= 1 step(s) to forward and update"

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state = deepcopy(self.model.state_dict())
        self.tent_optimizer_state = deepcopy(self.tent_optimizer.state_dict())

        # batch sizes
        self.norm_batch_size = norm_batch_size
        self.tent_batch_size = tent_batch_size

    def forward(self, x):
        # forward with norm adaptation
        self.configure_norm()
        with torch.no_grad():
            for minibatch in torch.split(x, self.norm_batch_size):
                self.model(minibatch)

        # forward with tent adaptation
        self.configure_tent()
        tent.check_model(self.model)
        for minibatch in torch.split(x, self.tent_batch_size):
            for _ in range(self.tent_steps):
                tent.forward_and_adapt(minibatch, self.model, self.tent_optimizer)

        # now that we learned from tent, do a final forward pass to get outputs
        self.configure_output_pass()
        outputs = []
        with torch.no_grad():
            for minibatch in torch.split(x, 1024):
                out = self.model(minibatch)
                outputs.append(out)

        return torch.cat(outputs, dim=0)

    def reset(self):
        if self.model_state is None or self.tent_optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.model.load_state_dict(self.model_state, strict=True)
        self.tent_optimizer.load_state_dict(self.tent_optimizer_state)

    def configure_norm(self):
        """Configure model for norm adaptation."""
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # train mode to use batch statistics
                m.train()
                m.eps = self.norm_eps
                m.momentum = self.norm_momentum

    def configure_tent(self):
        """Configure model for tent adaptation."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                # keep batchnorms in train mode so that tent can use batch statistics (as usually done with tent)
                # but enable gradients for scale and shift parameters this time
                m.requires_grad_(True)

    def configure_output_pass(self):
        """Configure model for output pass without adaptation."""
        self.model.eval()
        self.model.requires_grad_(False)
