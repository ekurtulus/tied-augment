import argparse
import logging
import math
import os
import json
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy

logger = logging.getLogger(__name__)
best_acc = 0


def save_checkpoint(state, checkpoint, filename='checkpoint.pt'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def main(args):
    
    global best_acc

    def create_model(args):
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)

    logger.info(dict(args._get_kwargs()))

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)

    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, '../datasets')

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,
        sampler=train_sampler(labeled_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True)

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers * 2,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    model.to(args.device)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    if args.continue_training and os.path.isfile(os.path.join(args.out, "checkpoint.pt" )):
        checkpoint = torch.load(os.path.join(args.out, "checkpoint.pt" ))
        best_acc = checkpoint['best_acc']
        args.start_step = checkpoint['epoch'] * args.eval_step
        model.load_state_dict(checkpoint['state_dict'])
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        args.start_step = 0

    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    model.zero_grad()
    return train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)

class MeanWrapper(nn.Module):
    def __init__(self, criterion):
        super(MeanWrapper, self).__init__()
        self.criterion = criterion
        
    def forward(self, *args):
        return self.criterion(*args).mean()
    
class WeightScheduler:
    def __init__(self, args):
        self.args = args
        self.step = 0
        if args.similarity_schedule == "constant":
            self.values = [args.similarity_weight] * (args.total_steps)
        elif args.similarity_schedule == "random":
            self.values = np.random.uniform(-args.similarity_weight, args.similarity_weight, args.total_steps)
        elif args.similarity_schedule == "probabilistic":
            perm = np.random.permutation(args.total_steps)
            values = np.asarray([args.similarity_weight] * (args.total_steps))
            values[perm[:len(values) // 2]] *= -1
            self.values = values

        elif args.similarity_schedule == "warmup":
            values = np.linspace(0, args.similarity_weight,  ( args.total_steps) // 10).tolist()
            values += [args.similarity_weight] * ((args.total_steps) - (args.total_steps // 10))
            self.values = values

        elif args.similarity_schedule == "linear":
            self.values = np.linspace(0, args.similarity_weight, args.total_steps)
    
    def __call__(self):
        value = self.values[self.step]
        self.step += 1
        return value    
    
    
def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):

    similarity_fn = nn.CosineSimilarity() if args.tied_augment else nn.MSELoss(reduction="none")
    
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()

    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    weight_scheduler = WeightScheduler(args)

    model.train()
    
    metrics_cache = {"similarity" : [], 
                     "mask_probs" : [],
                     "train_loss" : [],
                     "train_loss_x" : [],
                     "train_loss_u" : [],
                     "test_accuracy" : [],
                     "mean_test_acc" : []
                    }
    
    for step in range(args.start_step, args.total_steps):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        mask_probs = AverageMeter()
        similarity_losses = AverageMeter()

        try:
            if args.similarity_loss_on == "unlabeled":
                inputs_x, targets_x = labeled_iter.next()
            else:
                (inputs_first, inputs_second), targets_x = labeled_iter.next()
                inputs_x = torch.cat((inputs_first, inputs_second), 0)
                targets_x = torch.cat((targets_x, targets_x), 0)
        except:

            if args.world_size > 1:
                labeled_epoch += 1
                labeled_trainloader.sampler.set_epoch(labeled_epoch)
            labeled_iter = iter(labeled_trainloader)
            inputs_x, targets_x = labeled_iter.next()

        try:
            (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
        except:
            if args.world_size > 1:
                unlabeled_epoch += 1
                unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
            unlabeled_iter = iter(unlabeled_trainloader)
            (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

        data_time.update(time.time() - end)
        
        batch_size = inputs_x.shape[0]
        inputs = interleave(
            torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 
                        2*args.mu+1 if args.similarity_loss_on == "unlabeled" else 2*args.mu+2).to(args.device)
        targets_x = targets_x.to(args.device)
        
        features, logits = model(inputs)
        logits = de_interleave(logits, 2*args.mu+1 if args.similarity_loss_on == "unlabeled" else 2*args.mu+2)
        logits_x = logits[:batch_size]
        logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
        del logits
        
        features_l_first, features_l_second = features[:batch_size].chunk(2)
        features_u_w, features_u_s = features[batch_size:].chunk(2)
        del features            
        Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
        
        pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
        max_probs, targets_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()
        Lu = (F.cross_entropy(logits_u_s, targets_u,
                              reduction='none') * mask).mean()
        
        if args.tied_augment:
            if args.similarity_loss_on == "unlabeled":
                first_features, second_features = features_u_w, features_u_s
                features_mask = mask
            else:
                first_features, second_features = features_l_first, features_l_second
                features_mask = torch.ones(first_features.shape[0], device=first_features.device,
                                           dtype=first_features.dtype)
            
            if not args.mask_similarity:
                similarity_loss = weight_scheduler() * similarity_fn(first_features, second_features).mean()
                loss = Lx + args.lambda_u * Lu - similarity_loss
            else:
                similarity_loss = similarity_fn(first_features, second_features)
                similarity_loss = weight_scheduler() * (similarity_loss * features_mask).mean()
                loss = Lx + args.lambda_u * Lu - similarity_loss
                
        else:
            similarity_loss = torch.tensor(0)
            loss = Lx + args.lambda_u * Lu
        if args.amp:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        losses.update(loss.item())
        losses_x.update(Lx.item())
        losses_u.update(Lu.item())
        similarity_losses.update(similarity_loss.item())
        optimizer.step()
        scheduler.step()
        if args.use_ema:
            ema_model.update(model)
        model.zero_grad()
        
        batch_time.update(time.time() - end)
        end = time.time()
        mask_probs.update(mask.mean().item())

        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model
        
        if step % args.eval_step == 0 and step != 0:
            epoch = step // args.eval_step
            logger.info('evaluating model for the %dth epoch' % epoch)
            if args.local_rank in [-1, 0]:
                test_loss, test_acc = test(args, test_loader, test_model, epoch)

                args.writer.add_scalar('train/1.train_loss', losses.avg, epoch)
                args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
                args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
                args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
                args.writer.add_scalar('train/5.similarity', similarity_losses.avg, epoch)
                args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
                args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
                
                test_accs.append(test_acc)
                
                metrics_cache["similarity"].append(similarity_losses.avg)
                metrics_cache["mask_probs"].append(mask_probs.avg)
                metrics_cache["train_loss"].append(losses.avg)
                metrics_cache["train_loss_x"].append(losses_x.avg)
                metrics_cache["train_loss_u"].append(losses_u.avg)
                metrics_cache["test_accuracy"].append(test_acc)
                metrics_cache["mean_test_acc"].append(np.mean(test_accs[-20:]))
                
                
                is_best = test_acc > best_acc
                best_acc = max(test_acc, best_acc)

                model_to_save = model.module if hasattr(model, "module") else model
                if args.use_ema:
                    ema_to_save = ema_model.ema.module if hasattr(
                        ema_model.ema, "module") else ema_model.ema
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model_to_save.state_dict(),
                    'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, args.out)

                
                logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
                logger.info('Mean top-1 acc: {:.2f}'.format(
                    np.mean(test_accs[-20:])))
                logger.info('Mean mask probs: {:.2f}'.format(mask_probs.avg))
                logger.info('Mean train_loss: {:.2f}'.format(losses.avg))
                logger.info('Mean train_loss_x: {:.2f}'.format(losses_x.avg))
                logger.info('Mean train_loss_u: {:.2f}'.format(losses_u.avg))
                logger.info('Mean train similarity loss: {:.2f}\n\n'.format(similarity_losses.avg))

        

    if args.local_rank in [-1, 0]:
        args.writer.close()
    
    return test_accs, metrics_cache

def test(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    parser.add_argument('--total-steps', default=2**20, type=int,
                        help='number of total steps to run')
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--out', default='./experiments',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    
    # Tied-FixMatch algorithm

    parser.add_argument("--tied_augment", action="store_true")
    parser.add_argument("--similarity_loss_on", default="unlabeled", 
                                                choices=["labeled", "unlabeled"])
    parser.add_argument("--similarity_loss", default="l2")
    parser.add_argument("--similarity_weight", type=float, default=0)
    parser.add_argument("--similarity_schedule", default="constant")

    parser.add_argument("--mask_similarity", action="store_true")
    
    parser.add_argument("--outfile", default="results.json")
    parser.add_argument("--name", required=True)

    parser.add_argument("--first_transform")
    parser.add_argument("--second_transform")
    parser.add_argument("--continue_training", action="store_true")
    
    args = parser.parse_args()
               
    args.out = os.path.join(args.out, args.name)
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    open(os.path.join(args.out, "args.txt"), "w").write(str(args))
    
    args.out = os.path.join(args.out, "run_" + str(len(os.listdir(args.out)) - 1 ) )
    
        
    test_accs, metrics_cache = main(args)
    open(os.path.join(args.out, "metrics.json"), "w").write(json.dumps(metrics_cache))
    
    file = open(args.outfile, "a")
    file.write(json.dumps( {"task" : args.dataset, "num_labeled" : args.num_labeled,
                           "accuracy" : np.mean(test_accs[-20:]), "name" : args.name} ) + "\n" )
    file.close()
    
    
    