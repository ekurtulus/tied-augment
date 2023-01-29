import torch
import argparse
import os
import time
from handlers import *
from tqdm import tqdm
from evaluation import * 
from datasets import load_dataset
import numpy as np
import random
import pickle
import json
from functools import partial

def cycle(loader):
    while True:
        for data in loader:
            yield data

def _forward(model, first, second, single_forward=False):
    if args.single_forward:
        inputs = torch.cat((first, second), 0)
        out = model(inputs)
        first_features, second_features = torch.split(out[0], len(first), dim=0)
        first_logits, second_logits = torch.split(out[1], len(first), dim=0)
    else:
        first_features, first_logits = model(first)
        second_features, second_logits = model(second)    
    
    return first_logits, second_logits, first_features, second_features
            
def normal_step(batch, model, criterion, device, step, optimizer, args, sam=False, sam_step="first"):
    model.zero_grad()
    
    first, second, y = [i.to(device, non_blocking=True) for i in batch]
     
    if step == "tied-augment":
        first_logits, second_logits, first_features, second_features = _forward(model, first, 
                                                                                second, single_forward=args.single_forward)
        loss = criterion(first_logits, second_logits, first_features, second_features, y,
                                     ce=True, contrastive=True)
    elif step == "similarity-only":
        first_logits, second_logits, first_features, second_features = _forward(model, first, 
                                                                                second, single_forward=args.single_forward)
        loss = criterion(first_logits, second_logits, first_features, second_features, y,
                                     ce=False, contrastive=True)
    else:
        raise ValueError("Unknown step : " + step)
        
    loss.backward()
    if sam:
        if sam_step == "first":
            optimizer.first_step(zero_grad=True)
        if sam_step == "second":
            optimizer.second_step(zero_grad=True)
    else:
        optimizer.step()

    if type(first_features) is list and type(second_features) is list:
        return first_logits, second_logits, first_features[-1], second_features[-1], y, loss
    else:
        return first_logits, second_logits, first_features, second_features, y, loss
            

def sam_step(batch, model, criterion, device, step, optimizer, args):
    
    model.zero_grad()
    
    first, second, y = [i.to(device, non_blocking=True) for i in batch]

    optimizer.zero_grad()
    
    if args.sam_batch_handling:
        enable_running_stats(model)

    _first_logits, _second_logits, _first_features, _second_features, y, _loss = normal_step(batch, model, criterion, device,
                                        args.sam_first_step, optimizer, args, sam=True, sam_step="first")
    _loss.backward()        
    optimizer.first_step()
    
    if args.sam_batch_handling:
        disable_running_stats(model)
        
    first_logits, y, loss = normal_step(batch, model, criterion, device,
                                        args.sam_second_step, optimizer, args, 
                                        sam=True, sam_step="second")
    optimizer.second_step()
    
    if type(first_features) is list and type(second_features) is list:
        return first_logits, second_logits, first_features[-1], second_features[-1], y, loss
    else:
        return first_logits, second_logits, first_features, second_features, y, loss
    

def _check_nan_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            if (p.grad.data != p.grad.data).any():
                return True
    return False 

def train(args, train_set=None, test_set=None, clean_train_set=None,
                 model=None, model_ema=None, optimizer=None, 
                 scheduler=None, criterion=None, 
                 evaluator=None, train_metrics=None, test_metrics=None,  
                 checkpointer=None, device=None, iteration=0, 
                 no_checkpoint=False):
    
    global TEMP
    
    train_set = cycle(train_set)
    print("Start iteration : ", args.start_iteration, ", end iteration : ", args.iterations, flush=True)
    running_loss = []
    step_function = normal_step if not args.sam else sam_step
    
    for iteration in range(args.start_iteration, args.iterations + 1):
        batch = next(train_set)
        
        first_logits, second_logits, first_features, second_features, y, loss = step_function(batch, model, criterion, device, args.step_function, optimizer, args)
        running_loss.append(loss.cpu().item())         
        evaluator(first_logits, y)
        scheduler.step()

        if args.model_ema and iteration % args.model_ema_steps == 0:
            model_ema.update(model)   
        
        if False:
            for key, item in calculate_norms(model, first_logits, second_logits, first_features, second_features).items():
                train_metrics[key].append(item)
            
        train_metrics["stepwise-loss"].append(loss.cpu().item())

        if args.track_test and iteration in args.iterations_to_test:
            if args.iterations_to_test[0] == iteration:
                open( os.path.join(args.outpath, args.name, "test_track.json"), "w").close()
            
            writing_temp = {}
            for key, item in watch_under_transforms(iteration, model, test_set, args, device=device).items():
                writing_temp[key] = item
            
            temp_file = open( os.path.join(args.outpath, args.name, "test_track.json"), "a")
            temp_file.write( json.dumps(writing_temp) + "\n" ) 
            temp_file.close()
                
        
        if iteration % args.epoch_every == 0 or iteration == args.iterations:
            epoch_loss = np.mean(running_loss)
            running_loss = []
            train_metrics["epochwise-loss"].append(epoch_loss)

            metrics = evaluator.compute(reset=True)
            train_metrics["accuracy"].append(metrics["accuracy"])
            if not no_checkpoint:
                checkpointer.save(os.path.join(args.outpath, args.name, "latest.pt"), iteration)

            test_softmaxes, test_preds, test_labels, test_loss = predict(model.module if args.model_ema else model, test_set, device=device, halftensor=args.halftensor)
            test_acc = calculate_accuracy(test_preds, test_labels, args.evaluation_strategy, args.num_classes)
            
            test_metrics["accuracy"].append(test_acc)
            test_metrics["loss"].append(test_loss)

            print("Evaluation for the ", iteration // args.epoch_every, "th epoch :", 
                  "\n\tTrain accuracy : ", metrics["accuracy"],
                  "\n\tTrain loss : ", epoch_loss,
                  "\n\tLearning rate : ", scheduler.get_last_lr(), 
                  "\n\tTest accuracy : ", test_acc,
                  "\n\tTest loss : ", test_loss,
                sep="", flush=True)

    print("Running final evaluation...", flush=True)
    results = all_evaluate(model, clean_train_set, test_set, args.robustness_path, args.mean, args.std, device=device, halftensor=args.halftensor,
                           c_evaluation=(args.task in ["cifar10", "cifar100"]), strategy=args.evaluation_strategy, num_classes=args.num_classes )

    print("Final evaluation results : ", end="")
    for key, value in results.items():
        print("\n\t", key, " : ", value, sep="", end="", flush=True)
    print("\n")
    return results




def run(args):
    device = torch.device("cuda:0")
    
    train_set, test_set, clean_train_set = load_dataset(args)
    model, optimizer, criterion, scheduler, model_ema = handle_model(args, device)
    print("criterion : ", criterion, flush=True)
    
    train_metrics = { "stepwise-loss" : [],
                      "epochwise-loss" : [],
                      "accuracy" : [],
                      "grad-norm-1" : [],
                      "grad-norm-2" : [],
                      "grad-norm-inf" : [],
                      "weight-norm-1" : [],
                      "weight-norm-2" : [],
                      "weight-norm-inf" : [],
                     
                      "first-feature-norm-1": [],
                      "first-feature-norm-2": [],
                      "first-feature-norm-inf" : [],

                      "second-feature-norm-1" : [],
                      "second-feature-norm-2" : [],
                      "second-feature-norm-inf" : [],
                      
                      "logit-divergence" : [], 
                      "feature-cosine" : [],
                      "feature-l2" : [],
                      "feature-l1" : [],
                    }
    
    test_metrics = {"loss" : [], "accuracy" : []}
    
    args.__dict__["start_iteration"] = 1    

    print("Training model : ", args.model, " number of parameters : ", f"{sum(i.numel() for i in model.parameters()):,}", flush=True)
    
    if args.continue_training and os.path.isfile(os.path.join(args.outpath, args.name, "latest.pt")):
        model, model_ema, train_metrics, test_metrics, \
        optimizer, scheduler, start_iteration = load_from_file(os.path.join(args.outpath, args.name, "latest.pt"),
                                                                   model=model, model_ema=model_ema, optimizer=optimizer,
                                                                   scheduler=scheduler)
        args.__dict__["start_iteration"] = start_iteration
                                   
    checkpointer = CheckpointHandler(os.path.join(args.outpath, args.name, "latest.pt"), 
                                     model=model, model_ema=model_ema, optimizer=optimizer, 
                                     scheduler=scheduler, train_metrics=train_metrics, 
                                     test_metrics=test_metrics) 
    
    evaluator = RunningEvaluator(args.num_classes, args.evaluation_strategy)
    
    
    if args.track_test:
        if args.step_finder_scheduler == "linear":
            args.__dict__["iterations_to_test"] = np.linspace(1, args.iterations, args.num_tests).astype('int') 
        elif args.step_finder_scheduler == "log":
            args.__dict__["iterations_to_test"] = np.geomspace(1, args.iterations, args.num_tests).astype('int')
    
    fn = partial(train, args,train_set=train_set, test_set=test_set, clean_train_set=clean_train_set,
                             model=model, model_ema=model_ema, optimizer=optimizer, 
                             scheduler=scheduler, criterion=criterion, 
                             evaluator=evaluator, train_metrics=train_metrics, test_metrics=test_metrics,  
                             checkpointer=checkpointer, device=device, iteration=0, no_checkpoint=args.no_checkpoint)
    
    if args.debug:
        with torch.autograd.detect_anomaly():
            results = fn()
    else:
        results = fn()
    
    if not args.no_checkpoint:
        checkpointer.save(os.path.join(args.outpath, args.name, "final.pt"), args.iterations + 1) 

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The real and most logical way of supervised contrastive learning.")

    # required args
    parser.add_argument("--model", required=True)
    parser.add_argument("--datapath", required=True)
    parser.add_argument("--outpath", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--name", required=True)

    parser.add_argument("--outfile", type=str, default="results.json")

    parser.add_argument("--robustness-path")

    # tied-augment args
    parser.add_argument("--contrastive_loss", type=str, default="mse", choices=["cosine", "vicreg", "cka", "mse", "mae",
                                                                                   "l2-norm", "cosine-l2-norm"])
    parser.add_argument("--contrastive_weight", type=float, default=1)
    
    parser.add_argument("--vicreg_cov_weight", type=float, default=1 / 25)
    parser.add_argument("--vicreg_sim_weight", type=float, default=25 / 25)
    parser.add_argument("--vicreg_var_weight", type=float, default=25 / 25)

    parser.add_argument("--divergence_loss", action="store_true")
    parser.add_argument("--divergence_weight", type=float, default=1)
    parser.add_argument("--both_branches_supervised", action="store_true")

    parser.add_argument("--poly_loss", action="store_true")
    parser.add_argument("--poly_eps", type=float, default=1)
    
    parser.add_argument("--feature_layers", default="-1", type=lambda x: [int(i) for i in x.split(",")])
    parser.add_argument("--feature_layers_weights", default="1", type=lambda x: [float(i) for i in x.split(",")])
    
    # augmentation
    parser.add_argument("--first_augmentation")
    parser.add_argument("--second_augmentation")    
    parser.add_argument("--use_same_crop", action="store_true")

    # Optimizer args
    parser.add_argument("--opt", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=.9)
    
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--beta_1", type=float, default=.9)
    parser.add_argument("--beta_2", type=float, default=.999)
    
    parser.add_argument("--trust_clip", action="store_true")
    parser.add_argument("--always_adapt", action="store_true")
    parser.add_argument("--trust_coeff", type=float, default=0.001)
    
    parser.add_argument("--dampening", type=float, default=0)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--amsgrad", action="store_true")
    
    parser.add_argument("--rmsprop_decay", type=float, default=0.99)

    # LR Scheduling args
    parser.add_argument("--scheduler", choices=["cosine", "cosine-restarts", "multiplicative", "exponential", "none"], default="none")
    parser.add_argument("--mult_factor", type=float, default=.95)
    parser.add_argument("--t_0", type=int, default=-1)
    parser.add_argument("--t_mult", type=float, default=1)
    parser.add_argument("--warmup_steps", type=float, default=-1)

    # Training args
    parser.add_argument("--epochs", type=float, default=200)
    parser.add_argument("--num_iters", type=int, default=-1)
    
    parser.add_argument("--halftensor", action="store_true")
    parser.add_argument("--continue_training", action="store_true")
    
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=-1)
    
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)

    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--dropout", type=float, default=0.)
    parser.add_argument("--random_init", action="store_true")
    parser.add_argument("--freeze_layers", action="store_true")
    parser.add_argument("--run_n_times", type=int, default=1)

    # Model ema
    parser.add_argument("--model_ema", action="store_true")
    parser.add_argument("--model_ema_steps", type=int, default=1)
    parser.add_argument("--model_ema_decay", type=float, default=0.9)
    
    # Wide ResNet specific args
    parser.add_argument("--resnet_depth", type=int, default=28)
    parser.add_argument("--resnet_width", type=int, default=10)
    
    # SAM args
    parser.add_argument("--sam", action="store_true")
    parser.add_argument("--sam_first_step", default="tied-augment")
    parser.add_argument("--sam_second_step", default="tied-augment")
    
    parser.add_argument("--adaptive_sam", action="store_true")
    parser.add_argument("--sam_rho", type=float, default=0.05)
    parser.add_argument("--sam_batch_handling", action="store_true")
    
    # Misc arguments
    parser.add_argument("--cosine_schedule", default="constant")    
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--step_function", default="tied-augment")
    parser.add_argument("--single_forward", action="store_true")
    
    # Training watching
    parser.add_argument("--track_test", action="store_true")
    parser.add_argument("--transforms_to_track", default="./transforms/crop_hflip_randaug_n2_m19_cutout.pt,./transforms/crop_hflip_randaug_n2_m14_cutout.pt,./transforms/crop_hflip_randaug_n1_m2_cutout.pt,./transforms/crop_hflip.pt")
    parser.add_argument("--num_tests", default=1000, type=int)
    parser.add_argument("--step_finder_scheduler", default="log", choices=["log", "linear"])

    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    assert len(args.feature_layers) == len(args.feature_layers_weights), "the length of the args.feature_layers must match args.feature_layers_weights"  
    
    print("This training will use the following devices : ", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] )
    print("This run uses the following folder : ", os.path.join(args.outpath, args.name))
    print("Running with the following args: ", args)
    print("Running for ", args.run_n_times," times...", flush=True)
    

    if not os.path.isdir(os.path.join(args.outpath, args.name)):
        os.makedirs(os.path.join(args.outpath, args.name))
    
    with open(os.path.join(args.outpath, args.name, "args.txt"), "w") as data:
        data.write(str(args))
  
    with open(os.path.join(args.outpath, args.name, "args.pkl"), "wb") as data:
        pickle.dump(args, data)
    
    if args.test_batch_size == -1:
        args.test_batch_size = args.batch_size * 2
    
    name = args.name
    start_from = 0

    metrics = []
    
    """
    if args.task in json.load(open(args.outfile)) and args.name in json.load(open(args.outfile))[args.task]:
        exit("already done.")
    """
    
    if args.continue_training:
        for i in range(args.run_n_times):
            p = os.path.join(args.outpath, name, "run_" + str(i), "final.pt")
            if os.path.isfile(p):
                print("We already trained the", i, "th run!")
                checkpoint = torch.load(p)
                metrics.append(checkpoint["test_metrics"])
                start_from += 1
            else:
                break
            
    if start_from == args.run_n_times:
        print("It seems like training here is already completed...")
        exit()
    
    for i in range(start_from, args.run_n_times):
        print("\nRunning ", i, "th time...", flush=True)
        args.name = os.path.join(name, "run_" + str(i))
        if not os.path.isdir(os.path.join(args.outpath, args.name)):
            os.makedirs(os.path.join(args.outpath, args.name))    
        
        metric = run(args)    
        metrics.append(metric)
    
    print("Mean and std of all runs : ", np.mean([i["accuracy"] for i in metrics]), np.std([i["accuracy"] for i in metrics]))
    print("Metrics : ", end="")
    for key in metrics[0].keys():
        print("\n\t", key, " : ", np.mean([i[key] for i in metrics]), sep="", end="", flush=True)
    
    mean_metrics = {key : np.mean([i[key] for i in metrics]) for key in metrics[0].keys()}
    mean_metrics["accuracy_std"] = np.std([i["accuracy"] for i in metrics])
                                           
    outfile = open(args.outfile, "a")
    outfile.write(json.dumps({"task" : args.task, "name" : name, "eval" : mean_metrics}) + "\n")
    outfile.close()
    
    """
    out_file = json.load(open(args.outfile)) if os.path.isfile(args.outfile) else {} 
    if  args.task not in out_file:
        out_file[args.task] = {}
    
    out_file[args.task][name] = mean_metrics
    
    time.sleep(random.randint(5,20))
    
    with open(args.outfile, "w") as data:
        data.write(json.dumps(out_file)) 
    """
                   