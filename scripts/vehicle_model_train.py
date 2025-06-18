"""
Created on 2025년 5월 21일
train_dir 변수는 Train dataset이 있는 폴더이다.
valid_dir 변수는 Valid dataset이 있는 폴더이다.
efficientnet으로 훈련하여 모델을 저장한다.
@author:  윤경섭
"""
import os
from pathlib import Path
import shutil
from typing import List
from vehicle_model_dataset_prepare import run_vmodel_dataset_prepare
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
from PIL import Image

DATA_PREPARE = False
def main():

    train_ratio = 0.8
    src_dir = r'E:\윤경섭\상세차종_re-id_datasets'
    dst_dir = r'E:\윤경섭\vehicle_model'
    if DATA_PREPARE:   
        run_vmodel_dataset_prepare(src_dir, dst_dir, train_ratio)

    train_dir = Path(dst_dir,'train').resolve()
    valid_dir = Path(dst_dir,'valid').resolve()

    # Check if train_dir and valid_dir exist
    if not train_dir.exists() or not valid_dir.exists():
        print(f"Error: '{train_dir}' 또는 '{valid_dir}' 폴더가 존재하지 않습니다.")
        exit(1)

    # EfficientNet training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    num_epochs = 50
    num_classes = len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    image_size = 224
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = VehicleDataset(str(train_dir), transform=transform)
    valid_dataset = VehicleDataset(str(valid_dir), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    EPOCHS_HEAD = 10
    def set_trainable_layers(model, epoch, EPOCHS_HEAD=10):
        if epoch < EPOCHS_HEAD:
            # fc layer만 학습
            for name, param in model.named_parameters():
                if name.startswith('_fc'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            # block1 이후 전체 학습 (block1, block2, ..., _fc)
            for name, param in model.named_parameters():
                if name.startswith('_blocks.0'):
                    param.requires_grad = False
                else:
                    param.requires_grad = True
    patience = 7  # EarlyStopping patience
    early_stopping = EarlyStopping(patience=patience)
    best_val_acc = 0.0
    best_model_path = 'efficientnet_vehicle_model_best.pth'
    # Learning rate scheduling variables
    current_lr = 1e-4
    min_lr = 1e-7
    lr_patience = 3
    lr_counter = 0
    best_val_loss = None
    for epoch in range(num_epochs):
        set_trainable_layers(model, epoch, EPOCHS_HEAD)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=current_lr)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, valid_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated and saved at epoch {epoch+1} with val_acc={val_acc:.4f}")
        # Learning rate scheduling (after EPOCHS_HEAD)
        if epoch >= EPOCHS_HEAD:
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                lr_counter = 0
            else:
                lr_counter += 1
                if lr_counter >= lr_patience:
                    new_lr = max(current_lr * 0.95, min_lr)
                    if new_lr < current_lr:
                        print(f"Reducing learning rate from {current_lr:.8f} to {new_lr:.8f} at epoch {epoch+1}")
                        current_lr = new_lr
                    lr_counter = 0
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break
    # After training, rename best model file to include best_val_acc
    new_model_path = f"efficientnet_vehicle_model_best_vacc{best_val_acc:.4f}.pth"
    if os.path.exists(best_model_path):
        os.rename(best_model_path, new_model_path)
    
    # Save class names to vehicle_classes.txt in the same directory as the model
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    classes_txt_path = os.path.join(os.path.dirname(new_model_path), 'vehicle_classes.txt')
    with open(classes_txt_path, 'w', encoding='utf-8') as f:
        for class_name in class_names:
            f.write(class_name + '\n')
    
    print(f'Training complete. Best validation accuracy: {best_val_acc:.4f}. Model saved as {new_model_path}')
    print(f'Class names saved to {classes_txt_path}')

class VehicleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                for fname in os.listdir(class_path):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                        self.samples.append((os.path.join(class_path, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

class EarlyStopping:
    def __init__(self, patience=3, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

if __name__ == "__main__":
    main()

