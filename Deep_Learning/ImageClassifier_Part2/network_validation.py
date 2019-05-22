import torch

class NetworkValidation:


    def validation(model, loader, criterion, device):
        ''' Validation set scores,
            returns loss, accuracy
        '''
        
        loss = 0
        accuracy = 0

        #Change to device
        model.to(device)

        # Iterating over Loader
        for ii, (inputs, labels) in enumerate(loader):

            # Changing to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass to trought the model
            outputs = model.forward(inputs)

            # Calculating the loss
            loss += criterion(outputs, labels).item()

            # Calculating the probability
            ps = torch.exp(outputs)

            # Calculating the accuracy
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return loss, accuracy