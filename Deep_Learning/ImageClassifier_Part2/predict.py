import argparse

from network import Network
from image_process import Process
from network_utils import NetworkUtils

# Argument Parser
argp = argparse.ArgumentParser(description='predict-file')

argp.add_argument('input_img', default='flowers/test/1/image_06752.jpg', nargs='?', action='store', type = str)
argp.add_argument('checkpoint', default='checkpoint.pth', nargs='?', action='store',type = str)
argp.add_argument('--top_k', default=5, dest='top_k', action='store', type=int)
argp.add_argument('--category_names', dest='category_names', action='store', default='cat_to_name.json')
argp.add_argument('--device', default='cpu', action='store', dest='device')

# Parse Arguments
parg = argp.parse_args()

device = parg.device
k_outputs = parg.top_k
input_img_path = parg.input_img
checkpoint_path = parg.checkpoint

#Load the network
model = NetworkUtils.load_network(checkpoint_path)

# Predicted Results
probs, classes = Network.predict(input_img_path, model, device, topk=k_outputs)

# Print Results
for i in list(zip(probs, classes)):
    print(f'{i[1]} with a probability of {i[0]}')