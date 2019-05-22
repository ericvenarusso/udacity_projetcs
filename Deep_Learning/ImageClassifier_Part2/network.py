from image_process import Process
from network_utils import NetworkUtils
from network_validation import NetworkValidation

import numpy as np
from collections import OrderedDict

import torch
from torch import nn
from torchvision import models

class Network:
    
    
    def create_architecture(model_name, dropout, hidden_units):
        ''' Create a torch deeplearning architecture.
            return model
        '''

        if model_name == 'vgg16':
            model = models.vgg16(pretrained=True)
            input_features = model.classifier[0].in_features
        elif model_name == 'resnet18':
            model = models.resnet18(pretrained=True)
            input_features = model.fc.in_features
        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            input_features = model.classifier[1].in_features
            
        for param in model.parameters():
            param.requires_grad = False
            
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(dropout)),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        
        model.classifier = classifier
        
        return model
            
    def train(model, trainloader, validloader, criterion, optimizer, device, epochs, print_every=40):
        ''' Train a torch deep learning model.
        '''
        
        print('Initializing training...')
        steps = 0

        # Change to device
        model.to(device)

        for e in range(epochs):
            running_loss = 0
            model.train()

            # Iterating over the trainloader
            for ii, (inputs, labels) in enumerate(trainloader):
                steps += 1

                # Change to device
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Validation steps
                if steps % print_every == 0:

                    # Model Evaluation
                    model.eval()

                    # Turning off the gradients
                    with torch.no_grad():
                        valid_loss, accuracy = NetworkValidation.validation(model, validloader, criterion, device)

                    print("Epoch: {}/{}... |".format(e+1, epochs),
                          "Training Loss: {:.4f} |".format(running_loss/print_every),
                          "Validation Loss: {:.4f} |".format(valid_loss/len(validloader)),
                          "Validation Accuracy: {:.4f} |".format(accuracy/len(validloader)))

                    running_loss = 0

                    #Turn back to train mode
                    model.train()

    def predict(path, model, device, topk):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
            returns top_probabilities, top_classes
        '''

        # Process the image
        processed_img = Process.PIL_process(path)

        # Numpy to Pytorch tensor
        pytorch_tensor = torch.tensor(processed_img)
        pytorch_tensor = pytorch_tensor.float().to(device)

        # Removing RunTimeError for missing batch size - add batch size of 1 
        pytorch_tensor = pytorch_tensor.unsqueeze(0)

        #Model 
        model.eval().to(device)

        # Classify images
        log_probs = model.forward(pytorch_tensor)
        probs = torch.exp(log_probs)

        # Top k, probabilities and indices
        top_probs, top_labels = probs.topk(topk)

        #Desataching
        top_probs = np.array(top_probs.detach())[0]
        top_labels = np.array(top_labels.detach())[0]

        # Convert indices into classes
        classes = {val: key for key, val in model.class_to_idx.items()}

        # Top classes
        top_classes = [classes[i] for i in top_labels]
        
        return top_probs, NetworkUtils.map_class_names(top_classes, 'cat_to_name.json')