import os
import argparse

import torch

def parse_args():
    parser = argparse.ArgumentParser("D2 model converter")

    parser.add_argument("--source_model", default="", type=str, help="Path or url to the  model to convert")
    parser.add_argument("--output_model", default="", type=str, help="Path where to save the converted model")
    return parser.parse_args()

def main():
    args = parse_args()
    args.source_model='sam_vit_l_0b3195.pth'
    args.source_model = 'swin_tiny_patch4_window7_224.pth'
    if os.path.splitext(args.source_model)[-1] != ".pth":
        raise ValueError("You should save weights as pth file")
    #source_weights = torch.load(args.source_model, map_location=torch.device('cpu'))
    source_weights = torch.load(args.source_model, map_location=torch.device('cpu'))["model"]
    converted_weights = {}
    keys = list(source_weights.keys())
    
    prefix = 'backbone.bottom_up.'
    for key in keys:
        converted_weights[prefix+key] = source_weights[key]
    print('hello')
    print(os.getcwd())
    args.output_model='./output/swin_tiny_patch4_window7_224_d2.pth'
    torch.save(converted_weights, args.output_model)

if __name__ == "__main__":
    main()
    


