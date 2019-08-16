import torch
from torchvision import models
from torchvision import transforms
import argparse

from utility import utils
from visualizations.visualize_simplebackprop import SimpleBackpropagation
from visualizations.visualize_deconvolution import Deconvolution 
from visualizations.visualize_gradcam import Grad_CAM


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, default='./input_images', help='filepath to input image directory')
    parser.add_argument("-a", "--arch", type=str, required=True, help='Neural architecture to be used')
    parser.add_argument("-t", "--target-layer", type=str, required=True, help='The layer to be visualized')
    parser.add_argument("-k", "--topk", type=int, default=1, help='The top K categories to be visualized')
    parser.add_argument("-o", "--output-dir", type=str, default="./results", help='Directory to store the results')
    parser.add_argument("-m", "--method", type=str, default='gradcam', help='Method for generating visual explanation, e.g., gradcam, backprop, deconvolution')
    parser.add_argument("--cuda", default=True, help='GPU Flag for acceleration')
    
    args = parser.parse_args()
    
    assert args.arch in model_names, "Requested Neural Architecture is not available"
    
    if args.method == 'gradcam':
        Grad_CAM(args.input_dir, args.target_layer, args.arch, args.topk, args.output_dir, args.cuda)
    elif args.method == 'backprop':
        SimpleBackpropagation(args.input_dir, args.target_layer, args.arch, args.topk, args.output_dir, args.cuda)
    elif args.method == 'deconvolution':
        Deconvolution(args.input_dir, args.target_layer, args.arch, args.topk, args.output_dir, args.cuda)
    
