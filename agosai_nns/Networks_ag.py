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


class DeepFullyConnected(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        # dims: [-1, 50, 100, 150, 200, 150, 100, 50, 4]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(-1, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 150),
            torch.nn.ReLU(),
            torch.nn.Linear(150, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 4)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class DeepConvolutional(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        # dims: [-1, 4] (rest are convolutional layers)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 7 * 7, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 4)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class DeepRecurrent(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        # dims: [-1, 4] (RNN hidden layers inside)
        self.model = torch.nn.Sequential(
            torch.nn.RNN(input_size=-1, hidden_size=30, num_layers=3, batch_first=True),
            torch.nn.Flatten(),
            torch.nn.Linear(30, 4)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class DeepMLPWithDropout(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        # dims: [-1, 60, 120, 180, 240, 180, 120, 60, 4]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(-1, 60),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(60, 120),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(120, 180),
            torch.nn.ReLU(),
            torch.nn.Linear(180, 240),
            torch.nn.ReLU(),
            torch.nn.Linear(240, 180),
            torch.nn.ReLU(),
            torch.nn.Linear(180, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 60),
            torch.nn.ReLU(),
            torch.nn.Linear(60, 4)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class AdvancedConvolutional(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        # dims: [-1, 4] (mixed convolutional and linear layers)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 4)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class EnhancedDeepFullyConnected(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        # dims: [-1, 64, 128, 256, 512, 256, 128, 64, 4]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(-1, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class ComplexRecurrentConvolutional(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        # dims: [-1, 4] (mixed RNN and Convolutional layers)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 100),
            torch.nn.RNN(input_size=100, hidden_size=50, num_layers=2, batch_first=True),
            torch.nn.Flatten(),
            torch.nn.Linear(50, 4)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class DeepMLPWithBatchNorm(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        # dims: [-1, 80, 160, 320, 160, 80, 4]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(-1, 80),
            torch.nn.BatchNorm1d(80),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 160),
            torch.nn.BatchNorm1d(160),
            torch.nn.ReLU(),
            torch.nn.Linear(160, 320),
            torch.nn.BatchNorm1d(320),
            torch.nn.ReLU(),
            torch.nn.Linear(320, 160),
            torch.nn.BatchNorm1d(160),
            torch.nn.ReLU(),
            torch.nn.Linear(160, 80),
            torch.nn.BatchNorm1d(80),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 4)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class HybridConvolutionalRNN(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        # dims: [-1, 4] (Convolutional layers followed by RNN layers)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 7 * 7, 100),
            torch.nn.ReLU(),
            torch.nn.RNN(input_size=100, hidden_size=50, num_layers=2, batch_first=True),
            torch.nn.Flatten(),
            torch.nn.Linear(50, 4)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class AdvancedMLPMultipleActivations(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        # dims: [-1, 128, 256, 512, 256, 128, 64, 4]
        self.model = torch.nn.Sequential(
            torch.nn.Linear(-1, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.Sigmoid(),
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4)
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
