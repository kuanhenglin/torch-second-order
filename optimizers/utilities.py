from inspect import signature

import numpy as np
import torch


def sample_data(data, sample):
    inputs, labels = data
    indices = np.random.permutation(inputs.size(0))
    if type(sample) == int:  # sample represents number of samples
        indices = indices[:sample]
    elif type(sample) == float:  # sample represents fraction of total data
        indices = indices[:round(inputs.size(0) * sample)]
    return (inputs[indices], labels[indices])


def identity(arr, fill=None, grad=False):
    """Creates copy of tensor or array of tensors

    Parameters:
        arr (Tensor or list(Tensor) or tuple(Tensor)): input tensor or array of tensors

    Returns:
        (Tensor or list(Tensor) or tuple(Tensor)): tensor or array of tensors in the same
                                                    type and dimensions as arr
    """
    if grad:
        fn = torch.clone
    elif fill is not None:
        fn = lambda arr_: torch.full(arr_.shape, fill, dtype=arr_.dtype)
    else:
        fn = lambda arr_: torch.clone(arr_).detach()

    if type(arr) in (list, tuple):
        return type(arr)([fn(arr_).to(arr_.device) for arr_ in arr])
    else:  # assume type is Tensor
        return fn(arr).to(arr.device)


def mul_sum(a, b=None, sqrt=False):
    """Computes the reduced sum of the pairwise product of a and b

    Parameters:
        a (Tensor or tuple(Tensor)): tensor or array of tensors for product
        b (Tensor or tuple(Tensor)): tensor or array of tensors for product, if b = None,
                                     then b = a
        sqrt (bool): if True, return the square root of the result
    """
    if b is None:
        b = a
    if type(a) in (list, tuple):
        assert type(b) in (list, tuple)
        result = sum([torch.sum(torch.mul(a_, b_)) for a_, b_ in zip(a, b)])
    else:  # assume type is tensor
        assert type(b) not in (list, tuple)
        result = torch.sum(torch.mul(a, b))
    if sqrt:
        result.sqrt_()
    return result


def forward(fn, data):
    """Performs forward propagation

    Parameters:
        fn (tuple[Callable, Callable]): tuple of (forward_, obj)
            forward_ (Callable): forward/model function, can take one parameter (inputs) or
                                 two parameters (inputs, labels), the latter useful for when
                                 input feeding depends on labels
                                 (inputs: Tensor, labels: Tensor) -> Tensor
            obj (Callable): objective function, computes the loss
                            (outputs: Tensor, labels: Tensor) -> Tensor
        data (tuple[Tensor, Tensor]): tuple of (inputs, labels)

    Returns:
        (tuple[Tensor, Tensor]): tuple of forward/model outputs and loss (outputs, loss)
    """
    (forward_, obj), (inputs, labels) = fn, data
    num_params = len(signature(forward_).parameters)
    if num_params == 1:
        outputs = forward_(inputs)
    elif num_params == 2:
        outputs = forward_(inputs, labels)
    else:
        raise ValueError("The forward function can only have 1 parameter (inputs) or"
                         "2 parameters (inputs, labels).")
    # print(outputs.shape, labels.shape)
    loss = obj(outputs, labels)
    return outputs, loss


def minibatch(autodiff, fn, data, batch=None):
    """Computes minibatch for autodiff function

    Parameters:
        autodiff (Callable): autodiff function to evaluate with minibatches
                             (outputs: Tensor, loss: Tensor) -> tuple(Tensor)
                             if autodiff is None, then compute minibatched forward propagation
        fn (tuple[Callable, Callable]): tuple of (forward, obj)
            forward (Callable): forward/model function, can take one parameter (inputs) or
                                two parameters (inputs, labels), the latter useful for when
                                input feeding depends on labels
                                (inputs: Tensor, labels: Tensor) -> Tensor
            obj (Callable): objective function, computes the loss
                            (outputs: Tensor, labels: Tensor) -> Tensor
        data (tuple[Tensor, Tensor]): tuple of (inputs, labels)

    Returns:
        result (tuple[Tensor]): result of minibatch autodiff
        outputs (Tensor): outputs of the forward/model function fn[0]
        loss (Tensor): loss of the forward/model function via obj
    """
    inputs, labels = data
    result = None

    if batch is None or batch >= inputs.size(0):
        if autodiff is None:
            outputs, loss = forward(fn, data)
        else:
            with torch.enable_grad():
                outputs, loss = forward(fn, data)
                result = autodiff(outputs, loss)

    else:
        i = 0
        outputs, loss = [], torch.tensor(0.0, device=inputs.device)
        while i < inputs.size(0):
            inputs_ = inputs[i:i + batch]
            labels_ = labels[i:i + batch]
            if autodiff is None:
                outputs_, loss_ = forward(fn, (inputs_, labels_))
            else:
                with torch.enable_grad():
                    outputs_, loss_ = forward(fn, (inputs_, labels_))
                    result_ = autodiff(outputs_, loss_)
                if result is None:
                    result = result_
                else:
                    torch._foreach_add_(result, result_, alpha=inputs_.size(0) / inputs.size(0))
            loss.add_(loss_, alpha=inputs_.size(0) / inputs.size(0))
            outputs.append(outputs_)
            i += batch
        outputs = torch.cat(outputs, dim=0)

    return result, (outputs, loss)