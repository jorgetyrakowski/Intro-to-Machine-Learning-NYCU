import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from models.vgg_paper import VGGPaper
from dataset import get_dataloaders

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

class FERTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        self.model = VGGPaper(num_classes=7).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=config['learning_rate'],  # 0.01 
            momentum=0.9,                # 0.9 
            weight_decay=0.0001,         # 0.0001 
            nesterov=True                # Nesterov momentum 
        )
        
        # Learning rate scheduler 
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.75, 
            patience=5,    
            verbose=True
        )
        
        # Dataloaders
        self.train_loader, _ = get_dataloaders(
            data_dir=config['data_dir'],
            batch_size=config['batch_size'],  # 64 
            num_workers=config['num_workers']
        )
        
        self.best_acc = 0.0
        self.train_losses = []
        self.train_accs = []
        self.learning_rates = []
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["epochs"]}')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            bs, ncrops, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w)
            targets = torch.repeat_interleave(targets, repeats=ncrops)
            
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            acc = 100. * correct / total
            pbar.set_postfix({
                'loss': running_loss/(batch_idx+1),
                'acc': acc,
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        epoch_loss = running_loss/len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, acc):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'accuracy': acc,
            'best_accuracy': self.best_acc,
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'learning_rates': self.learning_rates
        }
        
        path = os.path.join(
            self.config['checkpoint_dir'],
            f'checkpoint_epoch{epoch}_acc{acc:.2f}.pth'
        )
        torch.save(state, path)
        logger.info(f'Checkpoint saved: {path}')
    
    def plot_metrics(self):
        plt.figure(figsize=(15, 5))
        
        plt.subplot(131)
        plt.plot(self.train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(132)
        plt.plot(self.train_accs)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True)
        
        plt.subplot(133)
        plt.plot(self.learning_rates)
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.grid(True)
        plt.yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join('plots', f'training_metrics.png'))
        plt.close()
    
    def train(self):
        logger.info(f'Starting training on {self.device}')
        logger.info(f'Config: {self.config}')
        
        for epoch in range(self.config['epochs']):
            train_loss, train_acc = self.train_epoch(epoch)
            
            self.scheduler.step(train_acc)
            
            if train_acc > self.best_acc:
                self.best_acc = train_acc
                self.save_checkpoint(epoch, train_acc)
            
            if (epoch + 1) % 10 == 0:
                self.plot_metrics()
            
            # Logging
            logger.info(f'\nEpoch {epoch+1}/{self.config["epochs"]}:')
            logger.info(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            logger.info(f'Best Train Acc: {self.best_acc:.2f}%')
            logger.info(f'Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')

def main():
    config = {
        'data_dir': 'data',
        'checkpoint_dir': 'checkpoints',
        'learning_rate': 0.01,      
        'batch_size': 64,           
        'num_workers': 4,
        'epochs': 300,             
    }
    
    trainer = FERTrainer(config)
    trainer.train()

if __name__ == '__main__':
    main()