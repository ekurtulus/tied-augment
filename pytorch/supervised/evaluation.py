import torch
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms as T
import numpy as np
import os
import glob
import json
from tqdm import tqdm
from losses import jensen_shannon

@torch.no_grad()
def watch_under_transforms(iteration, model, test_set, args, device=torch.device("cuda:0")):
    base_transform = test_set.dataset.transform
    
    cache = {os.path.basename(key) : {} for key in args.transforms_to_track.split(",")}
    
    for transform in args.transforms_to_track.split(","):
        loaded_transform = torch.load(transform)
        test_set.dataset.transform = loaded_transform
        
        softmaxes, preds, labels, loss, features = predict(model, test_set, device=device, return_features=True)
        
        cache[os.path.basename(transform)]["softmaxes"] = softmaxes
        cache[os.path.basename(transform)]["preds"] = preds
        cache[os.path.basename(transform)]["labels"] = labels
        cache[os.path.basename(transform)]["loss"] = loss
        cache[os.path.basename(transform)]["features"] = features
    
    test_set.dataset.transform = base_transform
    
    result = {}
    cosine, l2, l1 = nn.CosineSimilarity(), nn.MSELoss(), nn.L1Loss()
    
    for i in cache:
        for j in cache:
            key = i + "-" + j
            result[key] = {"softmax-divergence" : jensen_shannon( cache[i]["softmaxes"],  cache[j]["softmaxes"],).item(),
                    "cosine" : cosine(cache[i]["features"], cache[j]["features"]).mean().item(),
                    "l2" : l2(cache[i]["features"], cache[j]["features"]).mean().item(),
                    "l1" : l1(cache[i]["features"], cache[j]["features"]).mean().item(),
                    "pred-overlap" : (cache[i]["preds"] == cache[j]["preds"]).sum().item() / len(test_set.dataset),
                    "true-pred-overlap" : torch.logical_and( 
                                            (cache[i]["preds"] == cache[i]["labels"]),
                                            (cache[j]["preds"] == cache[j]["labels"])
                                          ).sum().item() / len(test_set.dataset),
                    "false-pred-overlap" : torch.logical_and( 
                                            (cache[i]["preds"] != cache[i]["labels"]),
                                            (cache[j]["preds"] != cache[j]["labels"])
                                          ).sum().item() / len(test_set.dataset)
            }
                
    return {"iteration" : iteration, **result}
    

@torch.no_grad()
def predict(model, dataset, device=torch.device("cuda:0"), halftensor=False, return_features=False):
    softmax_outputs, outputs, labels, loss, features = [], [], [], 0, []
    
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    
    for x,y in dataset:
        x = x.to(device)
        labels.append(y)
        with torch.cuda.amp.autocast(halftensor):
            if return_features:
                feat, out = model(x, return_features=return_features)
                features.append(feat.cpu())
            else:
                out = model(x)

            outputs.append(out.cpu())
            softmax_outputs.append(torch.softmax(out, 1).cpu() )
            loss += criterion(out, y.to(device)).cpu().item()
            
    model.train()
    preds = torch.cat(outputs, 0)
    softmaxes = torch.cat(softmax_outputs)
    if return_features:
        return softmaxes, preds.argmax(1), torch.cat(labels), loss / len(dataset.dataset), torch.cat(features, 0)
    else:
        return softmaxes, preds.argmax(1), torch.cat(labels), loss / len(dataset.dataset)

@torch.no_grad()
def robustness_evaluation(model, robustness_data_path, mean, std, halftensor=False, device=torch.device("cuda:0")):
    model.eval()
    results = {}
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    std = std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    
    all_labels = torch.tensor(np.load(os.path.join(robustness_data_path, "labels.npy")), dtype=torch.long)
    for severity in range(5):        
        labels = all_labels[severity*10_000:(severity+1)*10_000]
        cache = {}
        for transform in glob.glob(os.path.join(robustness_data_path, "*")):
            if transform.endswith("labels.npy"):
                continue
            data = torch.tensor(np.load(transform)[severity*10_000:(severity+1)*10_000], dtype=torch.float).permute(0, 3, 1,2).div_(255).sub_(mean).div_(std)
            test_data = DataLoader(TensorDataset(data, labels), batch_size=2048)
            logits, preds, labels, _ = predict(model, test_data, device=device, halftensor=halftensor)
            cache[os.path.splitext(os.path.basename(transform))[0]] = (preds == labels).sum().item() / len(labels)
        
        results[severity] = cache
    
    model.train()
    return results

def calculate_accuracy(preds, labels, strategy, num_classes):
    if strategy == "top-1":
        return (preds == labels).sum().item() / len(labels) * 100
    else:
        trues = torch.zeros(num_classes)
        falses = torch.zeros(num_classes)
        for i in range(num_classes):
            mask = (labels == i)
            num_examples = torch.sum(mask).item()
            correct = (preds[mask] == i).sum().item()
            trues[i] += correct
            falses[i] += (num_examples - correct)        
            
        return (trues / (trues + falses)).mean().item() * 100
    
def all_evaluate(model, train_set, test_set, robustness_data_path, mean, std, device=torch.device("cuda:0"), halftensor=False, c_evaluation=False, strategy="top-1", num_classes=None):
    
    logits, preds, labels, test_loss = predict(model, test_set, halftensor=halftensor, device=device)
    test_acc = calculate_accuracy(preds, labels, strategy, num_classes)
    calibration = compute_calibration(labels.numpy(), preds.numpy(), np.asarray([logits[j, i] for j, i in enumerate(labels)]) )
       
    clean_train_logits, clean_train_preds, clean_train_labels, clean_train_loss = predict(model, train_set, device=device, halftensor=halftensor)
 

    results =  {
        "accuracy"  : test_acc,
        "final_test_loss" : test_loss,
        "final_clean_train_loss" : clean_train_loss,
        "final_clean_train_accuracy" : calculate_accuracy(clean_train_preds, clean_train_labels, strategy, num_classes),
        "expected_calibration_error" : calibration["expected_calibration_error"] * 100,
        "average_confidence" : calibration["avg_confidence"],
    }
    
    if c_evaluation and robustness_data_path is not None:
        c_results = robustness_evaluation(model, robustness_data_path, mean, std, halftensor=halftensor, device=device)
        results = {**results, 
                   "c_results" : np.mean([np.mean([item2 for key2, item2 in item1.items()]) for key1, item1 in c_results.items()]) * 100,
                  }
                  
    return results



def compute_calibration(true_labels, pred_labels, confidences, num_bins=10):
    """Collects predictions into bins used to draw a reliability diagram.
    Arguments:
        true_labels: the true labels for the test examples
        pred_labels: the predicted labels for the test examples
        confidences: the predicted confidences for the test examples
        num_bins: number of bins
    The true_labels, pred_labels, confidences arguments must be NumPy arrays;
    pred_labels and true_labels may contain numeric or string labels.
    For a multi-class model, the predicted label and confidence should be those
    of the highest scoring class.
    Returns a dictionary containing the following NumPy arrays:
        accuracies: the average accuracy for each bin
        confidences: the average confidence for each bin
        counts: the number of examples in each bin
        bins: the confidence thresholds for each bin
        avg_accuracy: the accuracy over the entire test set
        avg_confidence: the average confidence over the entire test set
        expected_calibration_error: a weighted average of all calibration gaps
        max_calibration_error: the largest calibration gap across all bins
    """
    assert(len(confidences) == len(pred_labels))
    assert(len(confidences) == len(true_labels))
    assert(num_bins > 0)

    bin_size = 1.0 / num_bins
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    indices = np.digitize(confidences, bins, right=True)

    bin_accuracies = np.zeros(num_bins, dtype=np.float)
    bin_confidences = np.zeros(num_bins, dtype=np.float)
    bin_counts = np.zeros(num_bins, dtype=np.int)

    for b in range(num_bins):
        selected = np.where(indices == b + 1)[0]
        if len(selected) > 0:
            bin_accuracies[b] = np.mean(true_labels[selected] == pred_labels[selected])
            bin_confidences[b] = np.mean(confidences[selected])
            bin_counts[b] = len(selected)

    avg_acc = np.sum(bin_accuracies * bin_counts) / np.sum(bin_counts)
    avg_conf = np.sum(bin_confidences * bin_counts) / np.sum(bin_counts)

    gaps = np.abs(bin_accuracies - bin_confidences)
    ece = np.sum(gaps * bin_counts) / np.sum(bin_counts)
    mce = np.max(gaps)

    return { "accuracies": bin_accuracies, 
             "confidences": bin_confidences, 
             "counts": bin_counts, 
             "bins": bins,
             "avg_accuracy": avg_acc,
             "avg_confidence": avg_conf,
             "expected_calibration_error": ece,
             "max_calibration_error": mce }

class RunningEvaluator:
    def __init__(self, num_classes, strategy="top-1"):
        self.num_classes = num_classes
        self.strategy = strategy
        self.reset()
    
    def __call__(self, x,y):
        if self.strategy == "top-1":
            correct = (x.argmax(1) == y).sum().item()
            self.trues += correct
            self.falses += (len(y) - correct)
        else:
            preds = x.argmax(1)
            for i in range(self.num_classes):
                mask = (y == i)
                num_examples = torch.sum(mask).item()
                correct = (x[mask] == i).sum().item()
                self.trues[i] += correct
                self.falses[i] += (num_examples - correct)
           

    def compute(self, reset=False):
        acc = self.trues / (self.trues + self.falses)
        if self.strategy == "per-class-mean":
            acc = acc.mean()
        if reset:
            self.reset()
        return {"accuracy" : acc * 100}
                                      
    def reset(self):
        if self.strategy == "top-1":
            self.trues = 0
            self.falses = 0
        else:
            self.trues = np.zeros(self.num_classes)
            self.falses = np.zeros(self.num_classes)    
