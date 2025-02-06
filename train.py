#!/usr/bin/env python3
"""
train.py: Train a new network on a dataset of flowers and save the model as a checkpoint.

Usage Examples:
    python train.py flowers --save_dir checkpoints --arch vgg16 --learning_rate 0.001 \
           --hidden_units 4096 --epochs 5 --gpu
"""
import argparse
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

def get_input_args():
    """
    Returns command line arguments for training.
    """
    parser = argparse.ArgumentParser(
        description='Train a flower classification model.'
    )
    
    # Positional argument: data directory
    parser.add_argument('data_dir', type=str,
                        help='Directory containing the train/valid/test subfolders.')
    
    # Optional arguments
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save the checkpoint (default: current directory).')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='Model architecture (e.g., vgg16, alexnet). Default: vgg16.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer. Default: 0.001.')
    parser.add_argument('--hidden_units', type=int, default=4096,
                        help='Number of units in hidden layer. Default: 4096.')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs. Default: 5.')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available.')

    return parser.parse_args()


def main():
    # 1. Parse command-line arguments
    args = get_input_args()
    
    # 2. Prepare directories
    data_dir = args.data_dir
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir  = os.path.join(data_dir, 'test')
    
    # 3. Define data transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    }
    
    # 4. Create ImageFolder datasets
    image_datasets = {
        'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
        'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
        'test':  datasets.ImageFolder(test_dir,  transform=data_transforms['test'])
    }
    
    # 5. Create DataLoaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
        'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=False),
        'test':  torch.utils.data.DataLoader(image_datasets['test'],  batch_size=32, shuffle=False)
    }
    
    # 6. Choose a pre-trained model
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif args.arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        input_size = 9216
    else:
        raise ValueError(f"Unsupported architecture '{args.arch}'. Try 'vgg16' or 'alexnet'.")
    
    # Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # 7. Define a new classifier
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, args.hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(args.hidden_units, 102)),  # 102 flower classes
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    
    # 8. Training setup
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    epochs = args.epochs
    steps = 0
    print_every = 20

    # 9. Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()
        
        for inputs, labels in dataloaders['train']:
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Validation pass
            if steps % print_every == 0:
                valid_loss = 0.0
                accuracy = 0.0
                model.eval()
                
                with torch.no_grad():
                    for inputs_val, labels_val in dataloaders['valid']:
                        inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                        
                        logps_val = model.forward(inputs_val)
                        batch_loss = criterion(logps_val, labels_val)
                        valid_loss += batch_loss.item()
                        
                        ps = torch.exp(logps_val)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels_val.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(dataloaders['valid']):.3f}.. "
                      f"Validation accuracy: {accuracy/len(dataloaders['valid']):.3f}")
                
                running_loss = 0.0
                model.train()
    
    # 10. Save the checkpoint
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    checkpoint = {
        'architecture': args.arch,
        'classifier': model.classifier,
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx,
        'optimizer_state': optimizer.state_dict(),
        'epochs': epochs
    }
    
    save_path = os.path.join(args.save_dir, 'checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")

if __name__ == '__main__':
    main()
