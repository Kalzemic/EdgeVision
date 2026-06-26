import torch 
import torch.nn as nn
from EdgeVision import EdgeVision
from torchvision.ops import complete_box_iou_loss
from Data import DetectionDataset, Collate_fn
import albumentations as A
from torch.utils.data import DataLoader
import cv2
import os
import numpy as np
from DetectorLoss import DetectorLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from metrics import decode_predictions, decode_targets

device = 'cuda' if torch.cuda.is_available() else 'cpu'


batchsize = 8
B = 1
classes = 2
epochs = 50
backbone_lr = 1e-3
head_lr= 1e-3
l2lambda = 1e-4
pretrained = False

net = EdgeVision(num_classes=classes,B=B,pretrained=pretrained).to(device)
S = net.S
criterion = DetectorLoss(B=B,S=S,num_classes=classes)
optimizer = torch.optim.AdamW([
    {'params': net.backbone.parameters(), 'lr': backbone_lr},
    {'params': net.head.parameters(),     'lr': head_lr},
], weight_decay=1e-4)

def train(net: EdgeVision, train_loader: DataLoader, val_loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: DetectorLoss):

    train_losses = []
    val_losses = []
    best_loss =  float('inf')

    for epoch in range(epochs):
        print(f'Epoch: {epoch + 1}')

        #Train
        net.train()
        train_loss = 0.0
        for img, lbl in tqdm(train_loader):
            img = img.to(device)
            lbl = lbl.to(device)
            out = net(img)
            loss = criterion(out,lbl) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f'train loss: {train_loss}')
        train_losses.append(train_loss)


        #Validation
        val_loss = 0.0
        val_metric = MeanAveragePrecision(box_format='xyxy')
        net.eval()
        with torch.no_grad():
            for img, lbl in tqdm(val_loader):

                img = img.to(device)
                lbl = lbl.to(device)
            
                out =net(img)
                loss = criterion(out,lbl)
                val_loss += loss.item()
                preds = decode_predictions(out, S, B, classes)
                tgts = decode_targets(lbl, S, B, classes)
                val_metric.update(preds, tgts)
        
        val_loss /= len(val_loader)
        map_result = val_metric.compute()
        val_map = map_result['map'].item()           # mAP@[.5:.95]
        val_map50 = map_result['map_50'].item()

        print(f"val loss: {val_loss:.4f}  mAP: {val_map:.4f}  mAP50: {val_map50:.4f}")
        val_losses.append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(net.state_dict(),'EdgeVision.pt')

    return net, train_losses, val_losses


def plot_losses(train_losses, val_losses, save_path='losses.png'):
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs_range, val_losses, label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()

def main():
    train_data, val_data, train_labels, val_labels = getData('./data')
    
    train_loader , val_loader = getTorchLoaders(train_data, val_data, train_labels, val_labels)

    net.eval()
    with torch.no_grad():
        img, lbl = next(iter(val_loader))
        img = img.to(device)
        pred = net(img)
        decoded = decode_predictions(pred, S, B, classes, conf_thresh=0.1)
        print(decoded[0])  

    model, train_losses, val_losses = train(net,train_loader,val_loader,optimizer,criterion)
    plot_losses(train_losses,val_losses)


    





def getData(dir_path='.'):
    train_data_path = f'{dir_path}/images/train'
    train_labels_path = f'{dir_path}/labels/train'
    val_data_path = f'{dir_path}/images/val'
    val_labels_path = f'{dir_path}/labels/val'

    train_data = []
    val_data = []
    train_labels = []
    val_labels = []

    label_files = sorted(os.listdir(train_labels_path))

    for idx, label_file in enumerate(label_files):
        # if idx > 1000: break 

        base_name= os.path.splitext(label_file)[0]
        image_file = base_name + '.jpg'

        
        label_path = os.path.join(train_labels_path, label_file)
        label = np.loadtxt(open(label_path, 'rb'), dtype=np.float32)
        if label.ndim == 1:
            label = np.expand_dims(label, axis=0)
        
        
        img_path = os.path.join(train_data_path, image_file)
        if not os.path.exists(img_path):
            print(f"Warning: Image {image_file} not found for label {label_file}. Skipping.")
            continue 

        img = cv2.imread(img_path)
        img = cv2.cvtColor(cv2.resize(img,(224,224)),cv2.COLOR_BGR2RGB)
        
        train_data.append(img)
        train_labels.append(label)

    label_files = sorted(os.listdir(val_labels_path))

    for idx, label_file in enumerate(label_files):
        if idx > 1000: break 

        base_name= os.path.splitext(label_file)[0]
        image_file = base_name + '.jpg'


        label_path = os.path.join(val_labels_path, label_file)
        label = np.loadtxt(open(label_path, 'rb'), dtype=np.float32)
        if label.ndim == 1:
            label = np.expand_dims(label, axis=0)
        
        
        img_path = os.path.join(val_data_path, image_file)
        if not os.path.exists(img_path):
            print(f"Warning: Image {image_file} not found for label {label_file}. Skipping.")
            continue 

        img = cv2.imread(img_path)
        img = cv2.cvtColor(cv2.resize(img,(224,224)),cv2.COLOR_BGR2RGB)
        
        val_data.append(img)
        val_labels.append(label)
    
    
    return np.stack(train_data), np.stack(val_data), train_labels, val_labels



def getTorchLoaders(train_data, val_data, train_labels, val_labels):
    
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.1, scale=(0.7, 1.3), rotate=0, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True))

    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3, clip=True))
    train_dataset = DetectionDataset(
    	train_data,
    	[torch.tensor(lbl, dtype=torch.float32) for lbl in train_labels],
    	img_size=224,
    	transform=train_transform,
    )
    val_dataset = DetectionDataset(
    	val_data,
    	[torch.tensor(lbl, dtype=torch.float32) for lbl in val_labels],
    	img_size=224,
    	transform=val_transform,  
    )
    train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True, drop_last=True, collate_fn=Collate_fn)
    val_loader = DataLoader(val_dataset,batch_size=batchsize, shuffle=False, collate_fn=Collate_fn)

    return train_loader, val_loader




if __name__ == '__main__':
    main()
