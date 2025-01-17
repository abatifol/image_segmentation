import os
import cv2
import time
import random
import numpy as np
from glob import glob
from tqdm import tqdm
from operator import add
import gc
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score
import albumentations as A
from torchsummary import summary
import tifffile
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def multiclass_dice_score(pred, target, num_classes, smooth=1e-6):
    pred = pred.argmax(dim=1)  # Convert logits to class indices
    dice_scores = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        dice_scores.append(dice_score(pred_cls, target_cls, smooth))
    return sum(dice_scores) / num_classes

def dice_score(pred, target, smooth=1e-6):
    # Compute intersection and union
    intersection = (pred * target).sum()
    total = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (total + smooth)
    return dice

def iou_score(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    total = pred.sum() + target.sum()
    iou = (intersection + smooth) / (total - intersection + smooth)
    return iou

def calculate_metrics(y_true, y_pred, smooth=1e-6):
    # Binarize ground truth and predictions
    y_true_bin = (y_true > 0.5).float().view(-1).cpu().numpy().astype(np.uint8)
    y_pred_bin = (torch.sigmoid(y_pred) > 0.5).float().view(-1).cpu().numpy().astype(np.uint8)

    # Calculate metrics
    score_jaccard = jaccard_score(y_true_bin, y_pred_bin, average='binary')
    score_f1 = f1_score(y_true_bin, y_pred_bin, average='binary')
    score_recall = recall_score(y_true_bin, y_pred_bin, average='binary')
    score_precision = precision_score(y_true_bin, y_pred_bin, average='binary')
    score_acc = accuracy_score(y_true_bin, y_pred_bin)
    score_dice = dice_score(y_pred_bin, y_true_bin, smooth)
    score_iou = iou_score(y_pred_bin, y_true_bin, smooth)
    return {'jaccard': score_jaccard, 'f1':score_f1, 'recall':score_recall, 'precision':score_precision, 'acc':score_acc, 'dice': score_dice, 'iou':score_iou} #, 'roc_auc':roc_auc}

def evaluate_model(model, dataloader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # Get model predictions
            predictions = model(images)

            # For multiclass classification (more than 1 class)
            if predictions.shape[1] > 1:
                predictions = torch.argmax(predictions, dim=1)
            else:  # Binary classification (2 classes)
                predictions = (torch.sigmoid(predictions) > 0.5).float()

            # Flatten predictions and labels
            all_preds.append(predictions.view(-1).cpu())
            all_labels.append(masks.view(-1).cpu())  

    # Concatenate all predictions and labels into 1D tensors
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Ensure both are of integer type for evaluation metrics
    all_preds = all_preds.astype(int)
    all_labels = all_labels.astype(int)

    # Compute metrics
    f1 = f1_score(all_labels, all_preds, average='macro')  # 'macro' for multi-class or binary
    acc = accuracy_score(all_labels, all_preds)
    dice = dice_score(all_preds, all_labels)  

    return {"Dice": dice, "F1": f1, "Accuracy": acc}


def build_metrics_table_and_visualize(models, dataloader, device='cuda', num_samples=3, model_dir='/kaggle/working'):
    metrics = []
    for model_name, model in models.items():
        # Load the model
        model.load_state_dict(torch.load(f"{model_dir}/{model_name}_checkpoint.pth",weights_only=True))
        model.to(device).eval()

        # Evaluate the model
        model_metrics = evaluate_model(model, dataloader, device)
        metrics.append({"Model": model_name, **model_metrics})

    # Create and display metrics table
    metrics_df = pd.DataFrame(metrics)
    metrics_df['Model']=metrics_df['Model'].map({
        'unet':'UNet','segnet':'SegNet','residual_unet':'Residual UNet','usegnet_3skip':'U-SegNet 3 skip','usegnet_2skip':'U-SegNet 2 skip',
    'usegnet_1skip':'U-SegNet 1 skip','resusegnet':'Residual U-SegNet'})
    metrics_df.to_csv(f'{model_dir}/best_metrics_comparison.csv')
    print(metrics_df)


def plot_metrics(metrics_file, output_dir, model_name):
    # Load metrics from CSV
    df = pd.read_csv(metrics_file)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot each metric
    metrics = [col for col in df.columns if col not in ['epoch', 'train_loss', 'val_loss']]
    for metric in metrics:
        plt.figure(figsize=(8, 6))
        
        if f"train_{metric}" in df.columns and f"val_{metric}" in df.columns:
            plt.plot(df['epoch'], df[f'train_{metric}'], label=f'Train {metric}', marker='o')
            plt.plot(df['epoch'], df[f'val_{metric}'], label=f'Validation {metric}', marker='o')
        elif metric in df.columns:  # Single metric column
            plt.plot(df['epoch'], df[metric], label=f'{metric}', marker='o')
        
        plt.xlabel("Epoch")
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} over Epochs ({model_name})")
        plt.legend()
        plt.grid()
        
        # Save the plot
        plot_path = os.path.join(output_dir, f"{model_name}_{metric}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")    