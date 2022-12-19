import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
from torch.optim.optimizer import Optimizer
import math

class SAM(Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        
        
class Lars(Optimizer):
    """ LARS for PyTorch
    
    Paper: `Large batch training of Convolutional Networks` - https://arxiv.org/pdf/1708.03888.pdf
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate (default: 1.0).
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        trust_coeff (float): trust coefficient for computing adaptive lr / trust_ratio (default: 0.001)
        eps (float): eps for division denominator (default: 1e-8)
        trust_clip (bool): enable LARC trust ratio clipping (default: False)
        always_adapt (bool): always apply LARS LR adapt, otherwise only when group weight_decay != 0 (default: False)
    """

    def __init__(
        self,
        params,
        lr=1.0,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        trust_coeff=0.001,
        eps=1e-8,
        trust_clip=False,
        always_adapt=False,
        **kwargs,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            trust_coeff=trust_coeff,
            eps=eps,
            trust_clip=trust_clip,
            always_adapt=always_adapt,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = self.param_groups[0]['params'][0].device
        one_tensor = torch.tensor(1.0, device=device)  # because torch.where doesn't handle scalars correctly

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            trust_coeff = group['trust_coeff']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # apply LARS LR adaptation, LARC clipping, weight decay
                # ref: https://github.com/NVIDIA/apex/blob/master/apex/parallel/LARC.py
                if weight_decay != 0 or group['always_adapt']:
                    w_norm = p.norm(2.0)
                    g_norm = grad.norm(2.0)
                    trust_ratio = trust_coeff * w_norm / (g_norm + w_norm * weight_decay + eps)
                    # FIXME nested where required since logical and/or not working in PT XLA
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, trust_ratio, one_tensor),
                        one_tensor,
                    )
                    if group['trust_clip']:
                        trust_ratio = torch.minimum(trust_ratio / group['lr'], one_tensor)
                    grad.add_(p, alpha=weight_decay)
                    grad.mul_(trust_ratio)

                # apply SGD update https://github.com/pytorch/pytorch/blob/1.7/torch/optim/sgd.py#L100
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(grad).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(grad, alpha=1. - dampening)
                    if nesterov:
                        grad = grad.add(buf, alpha=momentum)
                    else:
                        grad = buf

                p.add_(grad, alpha=-group['lr'])

        return loss
        
        
class Lamb(Optimizer):
    """Implements a pure pytorch variant of FuseLAMB (NvLamb variant) optimizer from apex.optimizers.FusedLAMB
    reference: https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/Transformer-XL/pytorch/lamb.py
    LAMB was proposed in `Large Batch Optimization for Deep Learning: Training BERT in 76 minutes`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging (bool, optional): whether apply (1-beta2) to grad when
            calculating running averages of gradient. (default: True)
        max_grad_norm (float, optional): value used to clip global grad norm (default: 1.0)
        trust_clip (bool): enable LAMBC trust ratio clipping (default: False)
        always_adapt (boolean, optional): Apply adaptive learning rate to 0.0
            weight decay parameter (default: False)
    .. _Large Batch Optimization for Deep Learning - Training BERT in 76 minutes:
        https://arxiv.org/abs/1904.00962
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
            self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999), eps=1e-6,
            weight_decay=0.01, grad_averaging=True, max_grad_norm=1.0, trust_clip=False, always_adapt=False, **kwargs,):
        defaults = dict(
            lr=lr, bias_correction=bias_correction, betas=betas, eps=eps, weight_decay=weight_decay,
            grad_averaging=grad_averaging, max_grad_norm=max_grad_norm,
            trust_clip=trust_clip, always_adapt=always_adapt)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        device = self.param_groups[0]['params'][0].device
        one_tensor = torch.tensor(1.0, device=device)  # because torch.where doesn't handle scalars correctly
        global_grad_norm = torch.zeros(1, device=device)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')
                global_grad_norm.add_(grad.pow(2).sum())

        global_grad_norm = torch.sqrt(global_grad_norm)
        # FIXME it'd be nice to remove explicit tensor conversion of scalars when torch.where promotes
        # scalar types properly https://github.com/pytorch/pytorch/issues/9190
        max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device)
        clip_global_grad_norm = torch.where(
            global_grad_norm > max_grad_norm,
            global_grad_norm / max_grad_norm,
            one_tensor)

        for group in self.param_groups:
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']
            grad_averaging = 1 if group['grad_averaging'] else 0
            beta3 = 1 - beta1 if grad_averaging else 1.0

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            if bias_correction:
                bias_correction1 = 1 - beta1 ** group['step']
                bias_correction2 = 1 - beta2 ** group['step']
            else:
                bias_correction1, bias_correction2 = 1.0, 1.0

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.div_(clip_global_grad_norm)
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient valuesa
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=beta3)  # m_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)  # v_t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                update = (exp_avg / bias_correction1).div_(denom)

                weight_decay = group['weight_decay']
                if weight_decay != 0:
                    update.add_(p, alpha=weight_decay)

                if weight_decay != 0 or group['always_adapt']:
                    # Layer-wise LR adaptation. By default, skip adaptation on parameters that are
                    # excluded from weight decay, unless always_adapt == True, then always enabled.
                    w_norm = p.norm(2.0)
                    g_norm = update.norm(2.0)
                    # FIXME nested where required since logical and/or not working in PT XLA
                    trust_ratio = torch.where(
                        w_norm > 0,
                        torch.where(g_norm > 0, w_norm / g_norm, one_tensor),
                        one_tensor,
                    )
                    if group['trust_clip']:
                        # LAMBC trust clipping, upper bound fixed at one
                        trust_ratio = torch.minimum(trust_ratio, one_tensor)
                    update.mul_(trust_ratio)

                p.add_(update, alpha=-group['lr'])

        return loss



class RMSpropTF(Optimizer):
    """Implements RMSprop algorithm (TensorFlow style epsilon)
    NOTE: This is a direct cut-and-paste of PyTorch RMSprop with eps applied before sqrt
    and a few other modifications to closer match Tensorflow for matching hyper-params.
    Noteworthy changes include:
    1. Epsilon applied inside square-root
    2. square_avg initialized to ones
    3. LR scaling of update accumulated in momentum buffer
    Proposed by G. Hinton in his
    `course <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.
    The centered version first appears in `Generating Sequences
    With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing (decay) constant (default: 0.9)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_decay (bool, optional): decoupled weight decay as per https://arxiv.org/abs/1711.05101
        lr_in_momentum (bool, optional): learning rate scaling is included in the momentum buffer
            update as per defaults in Tensorflow
    """

    def __init__(self, params, lr=1e-2, alpha=0.9, eps=1e-10, weight_decay=0, momentum=0., centered=False,
                 decoupled_decay=False, lr_in_momentum=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(
            lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay,
            decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
        super(RMSpropTF, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSpropTF, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(p)  # PyTorch inits to zero
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p)

                square_avg = state['square_avg']
                one_minus_alpha = 1. - group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if group['decoupled_decay']:
                        p.mul_(1. - group['lr'] * group['weight_decay'])
                    else:
                        grad = grad.add(p, alpha=group['weight_decay'])

                # Tensorflow order of ops for updating squared avg
                square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)
                # square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)  # PyTorch original

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(group['eps']).sqrt_()  # eps in sqrt
                    # grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)  # PyTorch original
                else:
                    avg = square_avg.add(group['eps']).sqrt_()  # eps moved in sqrt

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    # Tensorflow accumulates the LR scaling in the momentum buffer
                    if group['lr_in_momentum']:
                        buf.mul_(group['momentum']).addcdiv_(grad, avg, value=group['lr'])
                        p.add_(-buf)
                    else:
                        # PyTorch scales the param update by LR
                        buf.mul_(group['momentum']).addcdiv_(grad, avg)
                        p.add_(buf, alpha=-group['lr'])
                else:
                    p.addcdiv_(grad, avg, value=-group['lr'])

        return loss

    
class ExponentialMovingAverage(nn.Module):

    def __init__(self, model, decay=0.9999, device=None):
        super(ExponentialMovingAverage, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)        


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
            