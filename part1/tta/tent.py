# Adapted from:
#   https://github.com/DequanWang/tent/blob/7d236b42a3020f488a75d041f31a3f4ef4a521ea/tent.py
#   https://github.com/DequanWang/tent/blob/7d236b42a3020f488a75d041f31a3f4ef4a521ea/cifar10c.py
from __future__ import annotations

import torch

from torch import nn
from torch import optim

from tta.base import TTAMethod
from copy import deepcopy


class Tent(TTAMethod):

    def __init__(
        self,
        model,
        episodic=False,
        optim_steps=1,
        optim_lr=1e-3,
        optim_method="Adam",
        optim_beta=0.9,
        optim_momentum=0.9,
        optim_dampening=0.0,
        optim_nesterov=True,
        optim_wd=0.0,
    ):
        model = configure_model(model)
        check_model(model)
        super().__init__(model)

        params, _ = collect_params(model)
        optimizer = setup_optimizer(
            params,
            optim_lr,
            optim_method,
            optim_beta,
            optim_momentum,
            optim_dampening,
            optim_nesterov,
            optim_wd,
        )
        self.optimizer = optimizer
        self.steps = optim_steps
        assert optim_steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = copy_model_and_optimizer(
            self.model, self.optimizer
        )

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.optimizer)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(
            self.model, self.optimizer, self.model_state, self.optimizer_state
        )


def setup_optimizer(
    params,
    optim_lr=1e-3,
    optim_method="Adam",
    optim_beta=0.9,
    optim_momentum=0.9,
    optim_dampening=0.0,
    optim_nesterov=True,
    optim_wd=0.0,
):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if optim_method == "Adam":
        return optim.Adam(
            params,
            lr=optim_lr,
            betas=(optim_beta, 0.999),
            weight_decay=optim_wd,
        )
    elif optim_method == "SGD":
        return optim.SGD(
            params,
            lr=optim_lr,
            momentum=optim_momentum,
            dampening=optim_dampening,
            weight_decay=optim_wd,
            nesterov=optim_nesterov,
        )
    else:
        raise NotImplementedError


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ["weight", "bias"]:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " "check which require grad"
    assert not has_all_params, (
        "tent should not update all params: " "check which require grad"
    )
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
