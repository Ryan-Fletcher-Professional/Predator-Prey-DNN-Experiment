"""
THIS FILE WRITTEN BY ADVAIT GOSAI, RYAN FLETCHER AND SANATH UPADHYA
"""

import torch
from Globals import *

dtype = DTYPE # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class CreatureNetwork:
    def __init__(self, hyperparameters):
        self.model = None
        self.optimizer = None

    def get_inputs(self, state_info):
        """
        :param state_info: See Environment.get_state_info() return specification.
        :return: [ 2d-array : [float forward force, float rightwards force], float clockwise rotation in [0,2pi) ]
        """
        # Transform into a 1d array with environment info first then info about all other relevent creatures.
        input = self.transform(state_info)
        scores, loss = self.train_part34(self.model, self.optimizer, state_info, input)
        # This goes elsewhere?
        # self.model.eval()
        # scores = None
        # with torch.no_grad():
        #     scores = self.model(input)            
        return [[scores[0].item(), scores[1].item()], scores[2].item()], loss
    
    def train_part34(self, model, optimizer, state_info, input):
        """
        Train a model on CIFAR-10 using the PyTorch Module API.
        
        Inputs:
        - model: A PyTorch Module giving the model to train.
        - optimizer: An Optimizer object we will use to train the model
        
        Returns: Nothing, but prints model accuracies during training.
        CURRENTLY STOLEN FROM HW2
        """
        model.train()  # put model to training mode
        model = model.to(device=device)  # move the model parameters to CPU/GPU
        
        scores = model(input)
        loss = self.loss(state_info)

        # Zero out all of the gradients for the variables which the optimizer
        # will update.
        optimizer.zero_grad()

        # This is the backwards pass: compute the gradient of the loss with
        # respect to each  parameter of the model.
        loss.backward()

        # Actually update the parameters of the model using the gradients
        # computed by the backwards pass.
        optimizer.step()

        # if t % print_every == 0:
        #     print('Iteration %d, loss = %.4f' % (t, loss.item()))
        #     check_accuracy_part34(loader_val, model)
        #     print()
        
        return scores, loss.item()


class CreatureFullyConnected(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = hyperparameters["dimensions"]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dims[0], dims[1]),
            torch.nn.Linear(dims[1], dims[2]),
            torch.nn.Linear(dims[2], dims[3]),
            torch.nn.Linear(dims[3], dims[4])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
class DeepFullyConnected(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = [12, 50, 100, 150, 200, 150, 100, 50, 4]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dims[0], dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[1], dims[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[2], dims[3]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[3], dims[4]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[4], dims[5]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[5], dims[6]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[6], dims[7]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[7], dims[8])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class DeepMLPWithDropout(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = [12, 60, 120, 180, 240, 180, 120, 60, 4]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dims[0], dims[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[1], dims[2]),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(dims[2], dims[3]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[3], dims[4]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[4], dims[5]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[5], dims[6]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[6], dims[7]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[7], dims[8])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class EnhancedDeepFullyConnected(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = [12, 64, 128, 256, 512, 256, 128, 64, 4]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dims[0], dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[1], dims[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[2], dims[3]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[3], dims[4]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[4], dims[5]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[5], dims[6]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[6], dims[7]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[7], dims[8])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class DeepMLPWithLayerNorm(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = [12, 80, 160, 320, 160, 80, 4]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dims[0], dims[1]),
            torch.nn.LayerNorm(dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[1], dims[2]),
            torch.nn.LayerNorm(dims[2]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[2], dims[3]),
            torch.nn.LayerNorm(dims[3]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[3], dims[4]),
            torch.nn.LayerNorm(dims[4]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[4], dims[5]),
            torch.nn.LayerNorm(dims[5]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[5], dims[6])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class AdvancedMLPMultipleActivations(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = [12, 128, 256, 512, 256, 128, 64, 4]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(dims[0], dims[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[1], dims[2]),
            torch.nn.Tanh(),
            torch.nn.Linear(dims[2], dims[3]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[3], dims[4]),
            torch.nn.Sigmoid(),
            torch.nn.Linear(dims[4], dims[5]),
            torch.nn.Tanh(),
            torch.nn.Linear(dims[5], dims[6]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[6], dims[7])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())


