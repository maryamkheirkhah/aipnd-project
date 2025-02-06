#!/usr/bin/env python3
"""
predict.py: Predict flower name from an image using a trained deep learning model.

Usage Examples:
    python predict.py /path/to/image checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
"""
import argparse
import torch
import numpy as np
import json
from PIL import Image
from torchvision import models

def get_input_args():
    """
    Returns command line arguments for prediction.
    """
    parser = argparse.ArgumentParser(description="Predict flower name from an image.")
    
    parser.add_argument('image_path', type=str,
                        help='Path to the image file.')
    parser.add_argument('checkpoint', type=str,
                        help='Path to the saved model checkpoint (checkpoint.pth).')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Return top K most likely classes. Default: 5.')
    parser.add_argument('--category_names', type=str, default=None,
                        help='JSON file mapping categories to real names.')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU if available.')
    
    return parser.parse_args()

def load_checkpoint(filepath):
    """
    Loads the checkpoint file and rebuilds the model.
    """
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['architecture']
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture '{arch}' in checkpoint.")
    
    # Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Numpy array.
    """
    img = Image.open(image_path)
    
    # Resize: shorter side = 256, maintain aspect ratio
    width, height = img.size
    if width < height:
        img = img.resize((256, int(256 * height / width)))
    else:
        img = img.resize((int(256 * width / height), 256))
    
    # Center crop 224x224
    new_width, new_height = img.size
    left = (new_width - 224) / 2
    top = (new_height - 224) / 2
    right = left + 224
    bottom = top + 224
    img = img.crop((left, top, right, bottom))
    
    # Convert to numpy, scale to [0,1]
    np_image = np.array(img) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Reorder dimensions (H x W x C) -> (C x H x W)
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def predict(image_path, model, top_k=5, device='cpu'):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    Returns the probabilities and the corresponding class labels.
    """
    model.eval()
    model.to(device)
    
    # Process image
    np_image = process_image(image_path)
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model.forward(tensor_image)
    
    ps = torch.exp(output)
    top_p, top_class = ps.topk(top_k, dim=1)
    
    top_p = top_p.cpu().numpy().squeeze()
    top_class = top_class.cpu().numpy().squeeze()
    
    # Invert class_to_idx
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_labels = [idx_to_class[c] for c in top_class]
    
    return top_p, top_labels

def main():
    # 1. Parse command-line arguments
    args = get_input_args()
    
    # 2. Load the checkpoint
    model = load_checkpoint(args.checkpoint)
    
    # 3. Determine device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    # 4. Predict
    top_p, top_classes = predict(args.image_path, model, args.top_k, device)
    
    # 5. If category_names provided, convert classes to real flower names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_flowers = [cat_to_name.get(c, 'Unknown') for c in top_classes]
    else:
        top_flowers = top_classes
    
    # 6. Print results
    print("Results:")
    for i in range(len(top_flowers)):
        print(f" {i+1}) Flower: {top_flowers[i]}  Probability: {top_p[i]:.3f}")

if __name__ == '__main__':
    main()
