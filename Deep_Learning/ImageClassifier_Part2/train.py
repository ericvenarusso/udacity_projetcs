from network_utils import NetworkUtils
from image_process import Process
from network import Network

import torch 
from torch import nn, optim

import argparse

# Arg Parser
argp = argparse.ArgumentParser(description='train-file')

argp.add_argument('data_dir', default='flowers', nargs='?', action='store', type=str)
argp.add_argument('--arch', default='vgg16', action='store', type=str)
argp.add_argument('--learning_rate', default=0.0007, action='store', type=float)
argp.add_argument('--dropout', default=0.03, action='store', type=float)
argp.add_argument('--hidden_units', default=400, action='store', type=int )
argp.add_argument('--epochs', default=3, action='store', type=int)
argp.add_argument('--device', default='cpu', action='store', type=str)

#Parse Args
parg = argp.parse_args()

arch = parg.arch
device = parg.device
dropout = parg.dropout
data_dir = parg.data_dir
train_epochs = parg.epochs
hidden_units = parg.hidden_units
learning_rate = parg.learning_rate

# Image Directories
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Transform Images
train_transforms = Process.transform_compose('train')
valid_transforms = Process.transform_compose('validation')
test_transforms  = Process.transform_compose('test')

# Load Images
train_data = Process.load_image(train_dir, train_transforms)
valid_data = Process.load_image(valid_dir, valid_transforms)
test_data  = Process.load_image(test_dir, test_transforms)

# Loaders
trainloader  = Process.loader('train', train_data)
validloader = Process.loader('validation', valid_data)
testloader  = Process.loader('test', test_data)

# Network
model = Network.create_architecture(arch, dropout, hidden_units)
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

# Train
Network.train(model, trainloader, validloader, criterion, optimizer, device, epochs=train_epochs)
                
# Save Model
NetworkUtils.save_network(model, hidden_units, learning_rate, arch,
                          optimizer, train_epochs, train_data.class_to_idx,
                          'checkpoint.pth')