import torch
import torch.nn as nn
from losses import CriterionHandler
import os 
import glob
from optimization import *
import json
import random
from models import WideResNet
import timm
from losses import jensen_shannon
from torchvision import models


def handle_optimizer(model, args):
    arguments = dict(lr=args.lr, weight_decay=args.weight_decay)
    
    if args.opt in ["AdamW", "Adam", "FusedAdam"]:
        if args.opt == "FusedAdam":
            import apex
        opt = getattr(torch.optim if args.opt != "FusedAdam" else apex.optimizers, args.opt)
        arguments["betas"] = (int(args.beta_1), int(args.beta_2))
        arguments["eps"] = args.eps
        arguments["amsgrad"] = args.amsgrad
    
    elif args.opt == "RAdam":
        opt = torch.optim.RAdam
        arguments["betas"] = (int(args.beta_1), int(args.beta_2))
        arguments["eps"] = args.eps        
        
    elif args.opt in ["SGD", "FusedSGD"]:
        if args.opt == "FusedSGD":
            import apex
        opt = getattr(torch.optim if args.opt != "FusedSGD" else apex.optimizers, args.opt)
        arguments["momentum"] = args.momentum
        arguments["dampening"] = args.dampening
        arguments["nesterov"] = args.nesterov
    
    elif args.opt in ["RMSprop", "RMSpropTF"]:
         opt = torch.optim.RMSprop if args.opt == "RMSprop" else RMSpropTF
         arguments["alpha"] = args.rmsprop_decay
         arguments["momentum"] = args.momentum
         arguments["eps"] = args.eps
         
    elif args.opt == "Lamb":
        opt = Lamb
        arguments["eps"] = args.eps
        arguments["trust_clip"] = args.trust_clip 
        arguments["always_adapt"] = args.always_adapt
    
    elif args.opt == "Lars":
        opt = Lars
        arguments["trust_coeff"] = args.trust_coeff
        arguments["eps"] = args.eps
        arguments["trust_clip"] = args.trust_clip 
        arguments["always_adapt"] = args.always_adapt
        arguments["momentum"] = args.momentum
        arguments["nesterov"] = args.nesterov
        arguments["dampening"] = args.dampening
                
    if args.sam:
        optimizer = SAM(model.parameters(), opt, rho=args.sam_rho, adaptive=args.adaptive_sam, **arguments)
    else:
        optimizer = opt(model.parameters(), **arguments)
    
    return optimizer

class Dummy:
    def __init__(self, x):
        self.x = x
    def __call__(self, a):
        return self.x

def handle_scheduler(optimizer, n_iters, args):
    base_optimizer = optimizer.base_optimizer if args.sam else optimizer
    if args.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(base_optimizer, Dummy(1))
        
    elif args.scheduler == "cosine":
        in_ = n_iters - args.warmup_steps if args.warmup_steps != -1 else n_iters
        print("Scheduler steps : ", n_iters, flush=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_optimizer, n_iters)
    
    elif args.scheduler == "cosine-restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(base_optimizer, args.t_0, T_mult=args.t_mult)
    
    elif args.scheduler == "multiplicative":
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(base_optimizer, Dummy(args.mult_factor))
    
    elif args.scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(base_optimizer, args.mult_factor)
    
    else:
        raise ValueError("Unknown scheduler : ", args.scheduler)
    
    if args.warmup_steps != -1:
        scheduler = GradualWarmupScheduler(base_optimizer, 1, args.warmup_steps, after_scheduler=scheduler)
    
    return scheduler

def handle_model(args, device):
    model = create_model(args)
    
    model = model.to(device)

    # Optimizer
    optimizer = handle_optimizer(model, args)
    
    # scheduler
    scheduler = handle_scheduler(optimizer, args.iterations, args)
    
    # criterion
    criterion = CriterionHandler(args, args.iterations).to(device)
    
    model_ema = None
    if args.model_ema:
        model_ema = ExponentialMovingAverage(model, decay=args.model_ema_decay, device=device)
    
    return model, optimizer, criterion, scheduler, model_ema

def get_children(model: torch.nn.Module):
    children = list(model.children())
    flatt_children = []
    if children == []:
        return model
    else:
        for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

def freeze_layers(model):
    children = get_children(model)
    for param in model.parameters():
        param.requires_grad = False
    for child in children:
        if isinstance(child, timm.models.layers.linear.Linear) or isinstance(child, nn.Linear):
            for param in child.parameters():
                param.requires_grad = True   
            
class TimmWrapper(nn.Module):
    def __init__(self, model, num_classes):
        super(TimmWrapper, self).__init__()
        self.model = model
        self.fc = nn.Linear(model.num_features, num_classes)
    
    def forward(self, x, return_features=False):
        if self.training or return_features:
            features = self.model(x)
            logits = self.fc(features)
            return features, logits
        else:
            return self.fc(self.model(x))


class ResnetWrapper(nn.Module):
    def __init__(self, model, num_classes):
        super(ResnetWrapper, self).__init__()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x, return_features=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        features = torch.flatten(x, 1)
        x = self.model.fc(features)
        if self.training or return_features:
            return features, x
        else:
            return x
        
def create_model(args):
    if args.model == "wide-resnet":
        model = WideResNet(args.resnet_depth, args.num_classes, widen_factor=args.resnet_width, dropRate=args.dropout).train()
    else:
        model = ResnetWrapper(models.resnet50(pretrained=True), args.num_classes).train()
        #model = TimmWrapper(timm.create_model(args.model, num_classes=0, pretrained=True), args.num_classes)
            
    if args.freeze_layers:
        freeze_layers(model)
    
    model.train()
    return model
        
        
    
class CheckpointHandler:
    def __init__(self, outpath, model=None, model_ema=None, 
                 optimizer=None, scheduler=None,
                 train_metrics=None, test_metrics=None):
        
        self.outpath = outpath
        self.model = model
        self.model_ema = model_ema

        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        
    def save(self, out_path, iteration):
        model_state_dict = self.model.module.state_dict() if isinstance(self.model, nn.DataParallel) else self.model.state_dict()
        out_dict = {"model" : model_state_dict,
                    "train_metrics" : self.train_metrics,
                    "test_metrics" : self.test_metrics,
                    "optimizer" : self.optimizer.state_dict(),
                    "scheduler" : self.scheduler.state_dict() if self.scheduler is not None else None,
                    "iteration" : iteration,
                    }
        if self.model_ema is not None:
            out_dict["model_ema"] = self.model_ema.state_dict()
        else:
            out_dict["model_ema"] = None
        
        torch.save(out_dict, out_path)

def _load_scalar_safe(checkpoint, key, default_value):
    try:
        return checkpoint[key]
    except:
        print("The given checkpoint does not include the key: ", key, "using default value : ", default_value, "instead.")
        return default_value

def load_from_file(file, model=None, model_ema=None, optimizer=None, scheduler=None, scaler=None):
    
    checkpoint = torch.load(file)
    
    try:
        model.load_state_dict(checkpoint["model"])
    except: 
        model.module.load_state_dict(checkpoint["model"])
           
    train_metrics = _load_scalar_safe(checkpoint, "train_metrics", 
                                      {"stepwise-loss" : [],
                                       "epochwise-loss" : [],
                                       "accuracy" : [],
                                       "grad-norm-1" : [],
                                       "grad-norm-2" : [],
                                       "grad-norm-inf" : [],
                                       "weight-norm-1" : [],
                                       "weight-norm-2" : [],
                                       "weight-norm-inf" : [],
                                       } )
                    
    test_metrics = _load_scalar_safe(checkpoint, "test_metrics", {"loss" : [], "accuracy" : []} )
    
    iteration = _load_scalar_safe(checkpoint, "iteration", 0)
    
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    
    if model_ema is not None:
        model_ema.load_state_dict(checkpoint["model_ema"])
    
    print("Model successfully loaded from checkpoint : ", file, flush=True)
    
    return model, model_ema, train_metrics, test_metrics, \
          optimizer, scheduler, iteration
    
           
def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d):
            module.backup_momentum = module.momentum
            module.momentum = 0
        
    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if ( isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) ) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

@torch.no_grad()
def calculate_norms(model, first_logits, second_logits, first_features, second_features):
    return {
            "grad-norm-1" : torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in model.parameters()]), 2).item(),
            "grad-norm-2" : torch.norm(torch.stack([torch.norm(p.grad.detach(), 1) for p in model.parameters()]), 1).item(),
            "grad-norm-inf" : torch.norm(torch.stack([torch.norm(p.grad.detach(), float("inf")) for p in model.parameters()]), float("inf")).item(),
            "weight-norm-1" : torch.norm(torch.stack([torch.norm(p.data.detach(), 2) for p in model.parameters()]), 2).item(),
            "weight-norm-2" : torch.norm(torch.stack([torch.norm(p.data.detach(), 1) for p in model.parameters()]), 1).item(),
            "weight-norm-inf" : torch.norm(torch.stack([torch.norm(p.data.detach(), float("inf")) for p in model.parameters()]), float("inf")).item(),
            "first-feature-norm-1":  torch.linalg.vector_norm(first_features, ord=1, dim=1).mean().item(),
            "first-feature-norm-2": torch.linalg.vector_norm(first_features, ord=2, dim=1).mean().item(),
            "first-feature-norm-inf" : torch.linalg.vector_norm(first_features, ord=float('inf'), dim=1).mean().item(),
            "second-feature-norm-1" : torch.linalg.vector_norm(second_features, ord=1, dim=1).mean().item(),
            "second-feature-norm-2" : torch.linalg.vector_norm(second_features, ord=2, dim=1).mean().item(),
            "second-feature-norm-inf" : torch.linalg.vector_norm(second_features, ord=float('inf'), dim=1).mean().item(),
            "logit-divergence" : jensen_shannon(first_logits, second_logits).item(), 
            "feature-cosine" : nn.functional.cosine_similarity(first_features, second_features).mean().item(),
            "feature-l2" : nn.functional.mse_loss(first_features, second_features).item(),
            "feature-l1" : nn.functional.l1_loss(first_features, second_features).item(),
           }
