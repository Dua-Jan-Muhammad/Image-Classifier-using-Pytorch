import argparse
import functions
from torchvision import datasets, transforms, models
from torch import nn, optim
import torch
import os
from PIL import Image
import json


def main():
    parser = argparse.ArgumentParser( description='Prediction Script')
    
    
#     parser.add_argument('--image_data_dir', action='store',
#                         required=True,
#                         default=None,
#                         help='Enter path to images data. eg: "./Flowers" ')
    
    parser.add_argument('--PTmodel', 
                        dest='pretrained_model', default='resnet50',
                        choices=['inception_v3','resnet50', 'resnet18', 'resnet34','resnet101'],
                        help='Enter model to use. eg:resnet50 or inception_v3.  this classifier can currently work with\ resnet architectures.')

    parser.add_argument('--checkpoint', 
                        dest='checkpoint', default='checkpoint.pth',
                        help='Enter location to save checkpoint in.')
    
    parser.add_argument('--image_path', 
                        dest='image_path', required=True,
                        default=None,
                        help='Enter a single image path')
    
    parser.add_argument('--learning_rate', dest='learning_rate', default = 0.004,
                       help='Enter a Learning Rate')
    
    parser.add_argument('--gpu', action="store_true", default=False,
                        help='Enter a Boolean Value, true or False. True will turn the GPU on.')
    
    args = parser.parse_args()
    print(args)
#     image_data_dir = args.image_data_dir
    PTmodel = args.pretrained_model
    checkpoint_path = args.checkpoint
    image_path = args.image_path
    learning_rate = float(args.learning_rate)
    gpu = args.gpu
    
    #getting pretrained model and optimizer
    model = getattr(models, PTmodel)(pretrained=True)
#     optimizer = optim.Adam(model.fc.parameters(), lr=float(learning_rate))
    #loding model
    loaded_model = functions.load_model(checkpoint_path,gpu, model)
    
    
    #making prediction
    top_probabilities,top_classes = functions.predict(image_path, model, gpu,topk=5, )
#     print("The Top Probabilities are:",top_probabilities)
    
    #category mapping
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

        
    for i in range(len(top_probabilities)):
        # Get class names
         class_names = [cat_to_name[class_idx] for class_idx in top_classes]
          # Print results
#     print('he top classes are:', class_names)
    index = top_probabilities.index(max(top_probabilities))
    
    top_k_dict = {class_names[i]: top_probabilities[i] for i in range(len(top_probabilities))}
    print("top K classes along with associated probabilities: " ,top_k_dict)
    
    
    print(f"This flower should be: {class_names[index]}, with probability of {top_probabilities[index]*100:.2f}%")
    
    
    
if __name__ == '__main__':
    main()