import numpy as np
from PIL import Image

import torch
from torchvision import datasets, transforms

class Process:

    
    def transform_compose(set_type):
        ''' Create a transform compose object,
            returns Transform Compose
        '''
        
        if set_type == 'train':    
            transform = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
            
        elif (set_type == 'test') or (set_type == 'validation'):
            transform = transforms.Compose([transforms.RandomRotation(256),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
            
        return transform
    
    def load_image(path, transform):
        ''' Load images,
            returns datasets ImageFolder
        '''

        return datasets.ImageFolder(path, transform=transform)
    
    def loader(set_type, data):
        ''' Create train,test, validation Data Loader,
            returns a Torch DataLoader
        '''

        if set_type == 'train':
            loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
        
        elif (set_type == 'test') or (set_type == 'validation'):
            loader = torch.utils.data.DataLoader(data, batch_size=32)
        
        return loader
    
    def PIL_process(path):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''

        size = 256, 256
        crop_size = 224

        im = Image.open(path)

        if im.width > im.height:
            im.thumbnail((20000,size[1]),Image.ANTIALIAS)
        else:
            im.thumbnail((size[0],20000),Image.ANTIALIAS)

        # Crop the image
        top_crop = (size[1] - crop_size) / 2
        left_crop = (size[0] - crop_size)/ 2
        right_crop = (left_crop + crop_size)
        bottom_crop = (top_crop + crop_size)

        image = im.crop((left_crop, top_crop, right_crop, bottom_crop))

        # Transform the image into a np array
        img_array = np.array(image)

        # Color channel between 1 and 0
        np_image = img_array / 255

        # Normalize the image
        means = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        normalized_image = (np_image - means) / std

        # Transpose the image to make the collor channel the first dimension
        image = normalized_image.transpose(2, 0, 1)

        return image