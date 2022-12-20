import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch import Tensor
import numpy as np

class Poly1CrossEntropyLoss(nn.Module):
    def __init__(self,
                 epsilon: float = 1.0,
                 reduction: str = "mean",
                 weight: Tensor = None):
        """
        Create instance of Poly1CrossEntropyLoss
        :param num_classes:
        :param epsilon:
        :param reduction: one of none|sum|mean, apply reduction to final loss tensor
        :param weight: manual rescaling weight for each class, passed to Cross-Entropy loss
        """
        super(Poly1CrossEntropyLoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits, labels):
        """
        Forward pass
        :param logits: tensor of shape [N, num_classes]
        :param labels: tensor of shape [N]
        :return: poly cross-entropy loss
        """
        labels_onehot = F.one_hot(labels, num_classes=logits.size(1)).to(device=logits.device,
                                                                           dtype=logits.dtype)
        pt = torch.sum(labels_onehot * F.softmax(logits, dim=-1), dim=-1)
        CE = F.cross_entropy(input=logits,
                             target=labels,
                             reduction='none',
                             weight=self.weight)
        poly1 = CE + self.epsilon * (1 - pt)
        if self.reduction == "mean":
            poly1 = poly1.mean()
        elif self.reduction == "sum":
            poly1 = poly1.sum()
        return poly1


class CKA(nn.Module):
    def __init__(self):
        super(CKA, self).__init__()
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=K.device)
        I = torch.eye(n, device=K.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def forward(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)


class VicReg(nn.Module):
    def __init__(self, sim_loss_weight: float = 1,
                       var_loss_weight: float = 1,
                       cov_loss_weight: float = 1.0 / 25,):
        
        super(VicReg, self).__init__()
        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight

    def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes mse loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: invariance loss (mean squared error).
        """
    
        return F.mse_loss(z1, z2)
    
    
    def variance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes variance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: variance regularization loss.
        """
    
        eps = 1e-4
        std_z1 = torch.sqrt(z1.var(dim=0) + eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + eps)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        return std_loss
    
    
    def covariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes covariance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: covariance regularization loss.
        """
    
        N, D = z1.size()
    
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (N - 1)
        cov_z2 = (z2.T @ z2) / (N - 1)
    
        diag = torch.eye(D, device=z1.device)
        cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
        return cov_loss
    
    
    def vicreg_loss_func(
        z1: torch.Tensor,
        z2: torch.Tensor,
        sim_loss_weight: float = 25.0,
        var_loss_weight: float = 25.0,
        cov_loss_weight: float = 1.0,
    ) -> torch.Tensor:
        """Computes VICReg's loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
            sim_loss_weight (float): invariance loss weight.
            var_loss_weight (float): variance loss weight.
            cov_loss_weight (float): covariance loss weight.
        Returns:
            torch.Tensor: VICReg loss.
        """
    
        sim_loss = self.sim_loss_weight * self.invariance_loss(z1, z2) if self.sim_loss_weight != 0 else 0 
        var_loss = self.var_loss_weight * self.variance_loss(z1, z2) if self.var_loss_weight != 0 else 0 
        cov_loss = self.cov_loss_weight * self.covariance_loss(z1, z2) if self.cov_loss_weight != 0 else 0 
    
        loss = sim_loss + var_loss +  cov_loss
        return loss

class L2Norm(nn.Module):
    def __init__(self):
        super(L2, self).__init__()
        self.l2 = nn.MSELoss(reduction="mean")
    
    def forward(self, first, second):
        first_normalized = first / torch.linalg.norm(first)
        second_normalized = second / torch.linalg.norm(second)
        return self.l2(first_normalized, second_normalized)

class CosineL2Norm(nn.Module):
    def __init__(self):
        super(CosineL2Norm, self).__init__()
        self.cosine = nn.CosineSimilarity() 
        self.l2 = L2Norm()
   
    def forward(self, first, second):
        return self.cosine(first, second) - self.l2(first, second)

              
class MeanWrapper(nn.Module):
    def __init__(self, criterion):
        super(MeanWrapper, self).__init__()
        self.criterion = criterion
        
    def forward(self, *args):
        return self.criterion(*args).mean()
        
def jensen_shannon(first_logits, second_logits):
    first_logits = torch.softmax(first_logits, 1)
    second_logits = torch.softmax(second_logits, 1)
    logp_mixture = torch.clamp(torch.stack([first_logits, second_logits]).mean(axis=0), 1e-7, 1).log()
    
    first = F.kl_div(logp_mixture, first_logits, reduction="batchmean")
    second = F.kl_div(logp_mixture, second_logits, reduction="batchmean")
    return (first + second) / 2
    
class SupervisedWrapper(nn.Module):
    def __init__(self, criterion, divergence=False, divergence_weight=1, both_branches_supervised=False):
        super(SupervisedWrapper, self).__init__()
        self.criterion = criterion
        self.divergence = divergence
        self.divergence_weight = divergence_weight
        self.both_branches_supervised = both_branches_supervised
        
    def forward(self, first_logits, second_logits, y):
        if self.both_branches_supervised:
            loss = self.criterion(torch.cat((first_logits, second_logits), 0), torch.cat((y,y), 0) )
        else:
            loss = self.criterion(first_logits, y)
        
        if self.divergence:
            divergence_loss = jensen_shannon(first_logits, second_logits)
            loss += self.divergence_weight * divergence_loss
        
        return loss

class ContrastiveLayerwiseWrapper(nn.Module):
    def __init__(self, args, loss):
        super(ContrastiveLayerwiseWrapper, self).__init__()
        self.layers = args.feature_layers
        self.layer_weights = args.feature_layers_weights
        self.loss = loss
        
    def forward(self, first_features, second_features):
        loss = 0
        for layer, weight in zip(self.layers, self.layer_weights):
            loss += self.loss(first_features[layer], second_features[layer]) * weight
        return loss
    
    
def contrastive_handler(args):
    if args.contrastive_loss == "cosine":
        contrastive = MeanWrapper(nn.CosineSimilarity())
    elif args.contrastive_loss == "mae":
        contrastive = nn.L1Loss(reduction="mean")
    elif args.contrastive_loss == "mse":
        contrastive = nn.MSELoss(reduction="mean")
    elif args.contrastive_loss == "cka":
        contrastive = CKA()
    elif args.contrastive_loss == "l2-norm":
        contrastive = L2Norm()
    elif args.contrastive_loss == "cosine-l2-norm":
        contrastive = CosineL2Norm()
    elif args.contrastive_loss == "vicreg":
        contrastive = VicReg(sim_loss_weight=args.vicreg_sim_weight,
                             var_loss_weight=args.vicreg_var_weight,
                             cov_loss_weight=args.vicreg_cov_weight,)    
    return ContrastiveLayerwiseWrapper(args, contrastive)

        
class CriterionHandler(nn.Module):
    def __init__(self, args, steps):
        super(CriterionHandler, self).__init__()
        self.args = args
        
        self.contrastive = contrastive_handler(args)
        
        supervised = nn.CrossEntropyLoss() if not args.poly_loss else Poly1CrossEntropyLoss(epsilon=args.poly_eps)
        self.supervised = SupervisedWrapper(supervised, 
                                            divergence=args.divergence_loss, 
                                            divergence_weight=args.divergence_weight, 
                                            both_branches_supervised=args.both_branches_supervised)
        
        self.contrastive_weight = args.contrastive_weight
        self.cosine_schedule = args.cosine_schedule
        self.step = 0
        self.steps = steps
        self.printed = False

        
    def cosine_weight_handler(self):
        if self.cosine_schedule == "constant":
            similarity_weight = self.contrastive_weight
        elif self.cosine_schedule == "random":
            similarity_weight = np.random.uniform(-self.contrastive_weight, self.contrastive_weight)
        elif self.cosine_schedule == "probabilistic":
            if np.random.rand() > 0.5:
                similarity_weight = self.contrastive_weight
            else:
                similarity_weight = -self.contrastive_weight
        
        elif self.cosine_schedule == "negate_even":
            if self.step % 2 == 0:
                similarity_weight = -self.contrastive_weight
            else:
                similarity_weight = self.contrastive_weight

        elif self.cosine_schedule == "negate_odd":
            if self.step % 2 == 0:
                similarity_weight = self.contrastive_weight
            else:
                similarity_weight = -self.contrastive_weight

        elif self.cosine_schedule == "warmup":
            if self.step > (self.steps) // 10:
                if not self.printed:
                    print("cosine warmup done !", flush=True)
                    self.printed=True
                similarity_weight = self.contrastive_weight
            
            similarity_weight = np.linspace(0, self.contrastive_weight, self.steps)[self.step]

        elif self.cosine_schedule == "linear":
            similarity_weight = np.linspace(0, self.contrastive_weight, self.steps)[self.step]

        self.step += 1
        return similarity_weight

        
    
    def forward(self, first_logits, second_logits, first_features, second_features, y, ce=True, contrastive=True):
        contrastive_weight = self.cosine_weight_handler()
        if ce and contrastive:
            return self.supervised(first_logits, second_logits, y) - self.contrastive_weight * self.contrastive(first_features, second_features)
        elif ce:
            return self.supervised(first_logits, second_logits, y) 
        elif contrastive:
            return -self.contrastive_weight * self.contrastive(first_features, second_features)
        
