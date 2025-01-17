from utils import calculate_metrics, epoch_time
from losses import DiceBCELoss
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score
import albumentations as A
from torchsummary import summary



def train_model(model, train_loader, val_loader,device, model_name, criterion=None, optimizer=None, lr=1e-4, num_epochs=50, patience=10, output_path="/kaggle/working"):

    checkpoint_path = f"{output_path}/{model_name}_checkpoint.pth"
    model = model.to(device)
    
    # Set Optimizer and Loss
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    if criterion is None:
        criterion = DiceBCELoss()
    best_dice = -1.*float("inf")
    early_stopping_counter = 0
    
    train_losses = []
    val_losses = []

    train_metrics = {
        'jaccard':[],
        'f1':[],
        'recall':[],
        'precision':[],
        'acc':[],
        'dice':[],
        'iou':[]
    }
    val_metrics = {
        'jaccard':[],
        'f1':[],
        'recall':[],
        'precision':[],
        'acc':[],
        'dice': [],
        'iou':[]
    }
    
    for epoch in range(num_epochs):
        start_time = time.time()
        valid_loss = 0.0
        train_loss = 0.0
            
        model.train()
        train_epoch_metrics = {key: 0.0 for key in train_metrics}
        
        for x, y in train_loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
    
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            for key, value in calculate_metrics(y, y_pred).items():
                train_epoch_metrics[key]+=value
    
        train_loss = train_loss/len(train_loader)
        train_losses.append(train_loss)
        for key, value in train_epoch_metrics.items():
            train_metrics[key].append(value/len(train_loader))
        
        model.eval()
        val_epoch_metrics = {key: 0.0 for key in val_metrics}
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
    
                y_pred = model(x)
                loss = criterion(y_pred, y)
                valid_loss += loss.item()
                for key, value in calculate_metrics(y, y_pred).items():
                    val_epoch_metrics[key]+=value
    
            valid_loss = valid_loss / len(val_loader)
            val_losses.append(valid_loss)
            for key, value in val_epoch_metrics.items():
                val_metrics[key].append(value/len(val_loader))
        
        # Saving the model
        if val_metrics['dice'][-1] > best_dice:
            print(f"Validation Dice score improved from {best_dice:2.4f} to {val_metrics['dice'][-1]:2.4f}. Saving checkpoint: {checkpoint_path}")
    
            best_dice = val_metrics['dice'][-1]
            torch.save(model.state_dict(), checkpoint_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step(valid_loss)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}")
        print(f"\tVal Acc: {val_metrics['acc'][-1]:.3f} | Val Dice: {val_metrics['dice'][-1]:.3f} | Val IOU: {val_metrics['iou'][-1]:.3f}")

        # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, len(train_losses) + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        **{f'train_{key}': train_metrics[key] for key in train_metrics},
        **{f'val_{key}': val_metrics[key] for key in val_metrics},
    })
    metrics_df.to_csv(f"{model_name}_metrics.csv", index=False)
        
    return model, train_losses, val_losses, train_metrics, val_metrics


    