import json

import torch
from torchvision import models

class NetworkUtils:
    
    
    def save_network(model, hidden_layers, learning_rate, architecture, optimizer, epochs, class_to_idx, path):
        ''' Save a pytorch Network.
        '''

        print('Saving Network...')

        # Update the class_to_idx
        model.class_to_idx = class_to_idx

        # Create a checkpoint dict
        checkpoint = {'input_size': (3, 224, 224),
                     'output_size': 102,
                     'hidden_layers': hidden_layers,
                     'learning_rate': learning_rate,
                     'architecture': architecture,
                     'classifier': model.classifier,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'epoch': epochs,
                     'class_to_idx': model.class_to_idx}

        # Save the checkpoint
        torch.save(checkpoint, 'checkpoint.pth')
        
    def load_network(filepath):
        ''' Load a pytorch Network.
            return network
        '''
        
        print('Loading Network...')

        # Load the saved model
        checkpoint = torch.load(filepath)

        # Download and initiate the pretrained model
        model = getattr(models, checkpoint['architecture'])(pretrained=True)

        # Freeze parameters so we don't backprop through them
        for param in model.parameters(): 
            param.requires_grad = False

        # Load saved checkpoints
        model.class_to_idx = checkpoint['class_to_idx']
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])

        return model
    
    def map_class_names(classes_to_map, path):
        ''' Map Classes.
            returns list of mapped classes
        '''

        with open(path, 'r') as f:
            classes = json.load(f)
        
        return [classes[i] for i in classes_to_map]