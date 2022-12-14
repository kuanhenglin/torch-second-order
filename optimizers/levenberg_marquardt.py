"""Levenberg-Marquardt optimizer via Hessian-free methods
"""

import time

import torch
from torch.optim.optimizer import Optimizer
from torch import autograd

import optimizers.utilities as utils


class LevenbergMarquardt(Optimizer):

    def __init__(self, params, sample=0.05, minibatch=None, damping=1.0,
                 damping_update=dict(method="nielsen"), weight_decay=0.0,
                 lcg=dict(), lr_factor=1e-2, **kwargs):
        # default value groups
        if damping_update["method"] == "marquardt":
            damping_update_ = dict(rho_boost=0.25, rho_drop=0.75, boost=1.5, drop=1.5)
        elif damping_update["method"] == "nielsen":
            damping_update_ = dict(beta=2.0, gamma=3.0, p=3.0)
        else:
            raise ValueError("Damping update method only supports \"marquardt\" and \"nielsen\".")
        damping_update_.update(damping_update)
        lcg_ = dict(tol=1e-1, max=256, momentum=0.95, precond=True)
        lcg_.update(lcg)

        defaults = dict(sample=sample, minibatch=minibatch, damping_update=damping_update_,
                        weight_decay=weight_decay, lcg=lcg_, lr_factor=lr_factor, **kwargs)
        super(LevenbergMarquardt, self).__init__(params, defaults)

        self._init_autodiff()
        self._init_lcg()

        # variables
        self.state["damping"] = torch.tensor(damping)
        if damping_update["method"] == "nielsen":
            self.state["v"] = torch.tensor(damping_update_["beta"])

    def _init_autodiff(self):

        group = self.param_groups[0]

        def grad(fn, data, params, weight_decay=True):
            """Computes minibatch gradient

            Parameters:
                fn (tuple[Callable, Callable]): tuple of (forward, obj)
                    forward (Callable): forward/model function, can take one parameter (inputs) or
                                        two parameters (inputs, labels), the latter useful for when
                                        input feeding depends on labels
                                        (inputs: Tensor, labels: Tensor) -> Tensor
                    obj (Callable): objective function, computes the loss
                                    (outputs: Tensor, labels: Tensor) -> Tensor
                data (tuple[Tensor, Tensor]): tuple of (inputs, labels)
                params (tuple[Tensor]): network parameters

            Returns:
                g (tuple[Tensor]): gradient of loss w.r.t. network parameters, d loss / d params
                outputs (Tensor): outputs of the forward/model function fn[0]
                loss (Tensor): loss of the forward/model function via obj
            """
            grad_ = lambda outputs, loss: autograd.grad(loss, params)
            g, (outputs, loss) = utils.minibatch(grad_, fn, data, batch=group["minibatch"])

            if weight_decay and group["weight_decay"] != 0.0:
                torch._foreach_add_(g, group["params"], alpha=group["weight_decay"])
            return g, (outputs, loss)

        @torch.enable_grad()
        def Jvp(fn, params, v, retain_graph=None):
            """Computes Jacobian vector product, effectively forward-mode automatic differentiation

            Parameters:
                fn (Tensor): network outputs
                params (tuple[Tensor]): network parameters
                v (tuple[Tensor]): vector, should have same dimensions as params
                retain_graph (bool): if False, free graph for Jvp

            Returns:
                Jv (Tensor): Jacobian-vector product, same dimension as fn
            """
            u = torch.zeros_like(fn, requires_grad=True)  # u is dummy vector, same dimensions as fn
            uJ = autograd.grad(fn, params, grad_outputs=u, create_graph=True)
            (Jv,) = autograd.grad(uJ, u, grad_outputs=v, retain_graph=retain_graph)
            return Jv

        @torch.enable_grad()
        def GGNvp(fn, loss, params, v, retain_graph=None):
            """Computes generalized Gauss-Newton matrix-vector product
            Works with arbitrary objective functions

            Parameters:
                fn (Tensor): network outputs
                loss (Tensor): network loss
                params (tuple[Tensor]): network parameters
                v (tuple[Tensor]): vector, should have same dimensions as params
                retain_graph (bool): if False, free graph for GNvp

            Returns:
                JBJv (tuple[Tensor]): generalized Gauss-Newton matrix-vector product, same
                                      dimensions as params
            """
            (J_loss,) = autograd.grad(loss, fn, create_graph=True)  # J_loss = d loss / d fn
            # notice that d J_loss / d params = (d J_loss / d fn) (d fn / d params)
            #                                 = (d^2 loss / d^2 fn) (d fn / d params) = BJ
            BJv = self._Jvp(J_loss, params, v=v)
            JBJv = autograd.grad(fn, params, grad_outputs=BJv, retain_graph=retain_graph)
            return JBJv

        @torch.enable_grad()
        def GNvp(fn, params, v, retain_graph=None):
            """Computes Gauss-Newton matrix-vector product
            Assumes objective function is sum of squares, more efficient than generalized version

            Parameters:
                fn (Tensor): network outputs
                loss (Tensor): network loss
                params (tuple[Tensor]): network parameters
                v (tuple[Tensor]): vector, should have same dimensions as params
                retain_graph (bool): if False, free graph for GNvp

            Returns:
                JJv (tuple[Tensor]): Gauss-Newton matrix-vector product, same dimensions as params
            """
            Jv = self._Jvp(fn, params, v=v)  # J = d fn / d params
            JJv = autograd.grad(fn, params, grad_outputs=Jv, retain_graph=retain_graph)
            # average over minibatch, notice that 2 J^T J = J^T B J for sum of squares objective
            torch._foreach_mul_(JJv, 2.0 / fn.size(0))
            return JJv

        def LMvp(fn, data, params, v, generalized=True, damping=True, weight_decay=True):
            """Computes minibatch Levenberg-Marquardt matrix-vector product

            Parameters:
                fn (tuple[Callable, Callable]): tuple of (forward, obj)
                    forward (Callable): forward/model function, can take one parameter (inputs) or
                                        two parameters (inputs, labels), the latter useful for when
                                        input feeding depends on labels
                                        (inputs: Tensor, labels: Tensor) -> Tensor
                    obj (Callable): objective function, computes the loss
                                    (outputs: Tensor, labels: Tensor) -> Tensor
                data (tuple[Tensor, Tensor]): tuple of (inputs, labels)
                params (tuple[Tensor]): network parameters
                v (tuple[Tensor]): vector, should have same dimensions as params
                generalized (bool): if True, G = J^T B J; if False, G = 2 J^T J
                                    these are numerically equivalent if the objective function is
                                    sum of squares
                damping (bool): if True, apply damping factor, computing (G + damping I) v
                weight_decay (bool): if True, apply weight decay, computing (G + c I) v

            Returns:
                Gv (tuple[Tensor]): Levenberg-Marquardt matrix-vector product, same dimensions as
                                    params, note that this is different from Gv in GNvp()
            """
            if generalized:
                GNvp = lambda outputs, loss: self._GGNvp(outputs, loss, params, v)
            else:
                GNvp = lambda outputs, loss: self._GNvp(outputs, params, v)
            Gv, (outputs, loss) = utils.minibatch(GNvp, fn, data, batch=group["minibatch"])

            if damping:
                torch._foreach_add_(Gv, v, alpha=self.state["damping"])
            if weight_decay and group["weight_decay"] != 0.0:
                torch._foreach_add_(Gv, v, alpha=group["weight_decay"])
            return Gv, (outputs, loss)

        def precond(fn, data, params, damping=True, weight_decay=True):
            """Computes minibatch preconditioner for LCG with LM matrix-vector product
            """

            def precond_minibatch(fn, loss, params):
                # diagonal of loss hessian, diag( d^2 loss / d^2 fn )
                (J_loss,) = autograd.grad(loss, fn, create_graph=True)
                J_loss_sum = J_loss.sum(dim=0)
                B = []
                for i in range(fn.size(1)):  # compute diagonal one fn dimension at a time
                    mask = torch.zeros(fn.size(1), device=fn.device)
                    mask[i] = 1.0  # isolate fn dimension
                    retain_graph = i < fn.size(1) - 1  # only retain graph if not at last dimension
                    (B_column,) = autograd.grad(J_loss_sum, fn, grad_outputs=mask,
                                                retain_graph=retain_graph)
                    B.append(B_column[:, i:i + 1])
                B = torch.cat(B, dim=1)  # diagonal of B
                B.mul_(fn.size(0))
                rand = ((torch.rand(*fn.shape) > 0.5).to(torch.float32) * 2 - 1).to(fn.device)
                JB_sqrt = autograd.grad(fn, params, grad_outputs=rand * B.sqrt())
                return JB_sqrt

            GN_diag = lambda outputs, loss: precond_minibatch(outputs, loss, params)
            G_diag, (outputs, loss) = utils.minibatch(GN_diag, fn, data, batch=group["minibatch"])

            G_diag = [G_diag_.square() / data[0].size(0) for G_diag_ in G_diag]

            if damping:
                torch._foreach_add_(G_diag, self.state["damping"])
            if weight_decay and group["weight_decay"] != 0.0:
                torch._foreach_add_(G_diag, group["weight_decay"])
            return G_diag, (outputs, loss)

        self._grad, self._Jvp = grad, Jvp
        self._GGNvp, self._GNvp, self._LMvp = GGNvp, GNvp, LMvp
        self._precond = precond

    def _init_lcg(self):

        group = self.param_groups[0]

        @torch.no_grad()
        def lcg(A, b, x=None, tol=group["lcg"]["tol"], iter_max=group["lcg"]["max"], M=None):
            """Solves for x in the system Ax = b with linear conjugate gradient, assuming A is
            positive-semidefinite and dim(x) = dim(b), i.e., A is square

            Parameters:
                A (Callable): a function which computes Ax with A(x)
                b (tuple(Tensor)): right hand side of Ax = b
                x (tuple(Tensor)): what x is initialized to for LCG, defaults to 0
                tol (float): error tolerance of LCG, LCG breaks when |r| < tol
                iter_max (int): maximum number of iterations of LCG
                M (Callable): an (optional) function which preconditions r with M(r)

            Returns:
                x (tuple(Tensor)): solution to the system Ax = b
                r (tuple(Tensor)): residual of LCG, error vector
                i (int): number of LCG iterations
            """
            if x is None:  # default initialize x to 0
                x = utils.identity(b, fill=0.0)
                r = utils.identity(b)
            else:
                r = torch._foreach_sub(b, A(x))
            rr = utils.mul_sum(r)

            if M is None:
                p = utils.identity(r)
                for i in range(iter_max):
                    with torch.enable_grad():
                        Ap = A(p)
                    pAp = utils.mul_sum(p, Ap)
                    alpha = rr / pAp
                    torch._foreach_add_(x, p, alpha=alpha)
                    torch._foreach_sub_(r, Ap, alpha=alpha)
                    rr_next = utils.mul_sum(r)
                    if rr_next.sqrt() < tol:  # error is sufficiently small
                        break
                    beta = rr_next / rr
                    p = torch._foreach_add(r, p, alpha=beta)
                    rr = rr_next

            else:
                z = M(r)
                p = utils.identity(z)
                rz = utils.mul_sum(r, z)
                for i in range(iter_max):
                    with torch.enable_grad():
                        Ap = A(p)
                    pAp = utils.mul_sum(p, Ap)
                    alpha = rz / pAp
                    torch._foreach_add_(x, p, alpha=alpha)
                    torch._foreach_sub_(r, Ap, alpha=alpha)
                    rr_next = utils.mul_sum(r)
                    if rr_next.sqrt() < tol:  # error is sufficiently small
                        break
                    z = M(r)
                    rz_next = utils.mul_sum(r, z)
                    beta = rz_next / rz
                    p = torch._foreach_add(z, p, alpha=beta)
                    rr = rr_next
                    rz = rz_next

            return x, r, i

        @torch.no_grad()
        def line_search(fn, data, loss):
            params = group["params"]
            grad_desc = utils.mul_sum(self.state["gradient"], self.state["descent"])
            step_size, step_size_prev = 1.0, 0.0
            while step_size > 1e-3:
                torch._foreach_add_(params, self.state["descent"], alpha=step_size - step_size_prev)
                _, (_, loss_update) = utils.minibatch(None, fn, data, batch=group["minibatch"])
                loss_update += (group["weight_decay"] / 2) * utils.mul_sum(params)
                if (loss_update < loss + group["lr_factor"] * step_size * grad_desc):
                    break
                step_size_prev = step_size
                step_size /= 2
            return step_size, loss_update

        @torch.no_grad()
        def update_damping(rho):
            if group["damping_update"]["method"] == "marquardt":
                if rho > group["damping_update"]["rho_drop"]:
                    self.state["damping"].div_(group["damping_update"]["drop"])
                elif rho < group["damping_update"]["rho_boost"]:
                    self.state["damping"].mul_(group["damping_update"]["boost"])
            elif group["damping_update"]["method"] == "nielsen":
                if rho > 0.0:
                    beta = group["damping_update"]["beta"]
                    gamma = group["damping_update"]["gamma"]
                    p = group["damping_update"]["p"]
                    self.state["damping"].mul_(max(
                        1 / gamma, 1.0 - (beta - 1) * pow(2 * rho.cpu().item() - 1, p)))
                    self.state["v"] = torch.tensor(group["damping_update"]["beta"])
                else:
                    self.state["damping"].mul_(self.state["v"])
                    self.state["v"].mul_(2.0)
            return rho > 0.0  # only update network parameters if reduction ratio is positive

        @torch.no_grad()
        def step(fn, data, loss_no_decay):
            params = group["params"]
            loss = loss_no_decay + (group["weight_decay"] / 2) * utils.mul_sum(params)
            step_size, loss_update = self._line_search(fn, data, loss)

            delta_loss = loss_update - loss
            grad_GN_desc = []
            for i, desc_ in enumerate(self.state["descent"]):
                grad_, res_ = self.state["gradient"][i], self.state["lcg_residual"][i]
                grad_GN_desc.append(
                    grad_ + 0.5 * step_size * (-grad_ - self.state["damping"] * desc_ + res_))
            delta_quad = step_size * utils.mul_sum(self.state["descent"], grad_GN_desc)
            rho = delta_loss / delta_quad  # reduction ratio, actual / approximation

            if not update_damping(rho):  # undo network parameter update
                torch._foreach_sub_(params, self.state["descent"], alpha=step_size)

            del self.state["gradient"]
            del self.state["lcg_residual"]

        self._lcg = lcg
        self._line_search, self._step = line_search, step

    def step(self, fn, data):
        if len(self.param_groups) != 1:
            raise ValueError("Only one parameter group is currently supported.")

        outputs, loss = self.descent(fn, data)
        self._step(fn, data, loss_no_decay=loss)
        return outputs, loss

    @torch.no_grad()
    def descent(self, fn, data):
        group = self.param_groups[0]
        params = group["params"]

        self.state["gradient"], (outputs, loss) = self._grad(fn, data, params)

        if "descent" in self.state:
            tol = group["lcg"]["tol"] * utils.mul_sum(self.state["descent"], sqrt=True)
        else:  # initialize descent to all zeros
            self.state["descent"] = utils.identity(params, fill=0.0)
            tol = 1e-2 * group["lcg"]["tol"] * utils.mul_sum(self.state["gradient"], sqrt=True)
        torch._foreach_mul_(self.state["descent"], -group["lcg"]["momentum"])

        # use subsampled Gauss-Newton matrix (for performance and memory efficiency)
        data_sample = utils.sample_data(data, sample=group["sample"])
        if group["lcg"]["precond"]:
            precond, _ = self._precond(fn, data, params)
            torch._foreach_reciprocal_(precond)
            precond_fn = lambda r: torch._foreach_mul(r, precond)
        else:
            precond_fn = None
        self._precond(fn, data_sample, params, damping=False, weight_decay=False)
        LMvp = lambda v: self._LMvp(fn, data_sample, params, v)[0]
        _, self.state["lcg_residual"], lcg_iter =\
            self._lcg(LMvp, self.state["gradient"], self.state["descent"], tol=tol, M=precond_fn)

        torch._foreach_mul_(self.state["descent"], -1.0)  # we are solving (G + lambda I) d = -g
        return outputs, loss