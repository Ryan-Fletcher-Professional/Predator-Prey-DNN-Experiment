"""
THIS FILE WRITTEN BY RYAN FLETCHER AND SANATH UPADHYA
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
        scores = self.train_part34(self.model, self.optimizer, state_info, input)
        # This goes elsewhere?
        # self.model.eval()
        # scores = None
        # with torch.no_grad():
        #     scores = self.model(input)            
        return [[scores[0].item(), scores[1].item()], scores[2].item()]
    
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
        
        return scores


class CreatureFullyConnected(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = hyperparameters["dimensions"]
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
            torch.nn.Linear(dims[7], dims[8]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[8], dims[9]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[9], dims[10]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[10], dims[11]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[11], dims[12]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[12], dims[13]),
            torch.nn.ReLU()
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class CreatureFullyConnected2(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = hyperparameters["dimensions"]
        # [dims are as follows: -1,30,30,20,20,10,10,4]
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
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dims[5], dims[6]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[6], dims[7])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())
class Flatten(torch.nn.Module):
    def forward(self, x):
        N, C, H, W = x.size() # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image

class CreatureFullyConnected3(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        #dims = hyperparameters["dimensions"]
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 5, 1),
            torch.nn.LeakyReLU(0.01),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(32, 64, 5, 1),
            torch.nn.LeakyReLU(0.01),
            torch.nn.MaxPool2d(2,2),
            Flatten(),
            torch.nn.Linear(4 * 4 * 64, 1024),
            torch.nn.LeakyReLU(0.01),
            torch.nn.Linear(1024, 4)
    )
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
class CreatureFullyConnected4(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = hyperparameters["dimensions"]
        # [dims are as follows: -1,10,10,20,20,10,10,4]
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
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dims[5], dims[6]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[6], dims[7])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class CreatureFullyConnected5(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = hyperparameters["dimensions"]
        # [dims are as follows: -1,10,10,20,20,30,30,4]
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
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dims[5], dims[6]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[6], dims[7])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class CreatureFullyConnected6(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = hyperparameters["dimensions"]
        # [dims are as follows: -1,30,30,10,10,30,30,4]
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
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dims[5], dims[6]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[6], dims[7])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class CreatureFullyConnected7(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = hyperparameters["dimensions"]

        self.model = torch.nn.Sequential(
            torch.nn.Linear(12, 1024),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 7*7*128),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm1d(6272),
            torch.nn.ReLU(),
            torch.nn.Linear(6272, 64),
            torch.nn.LeakyReLU(0.05),
            #torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64, 4)


        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class CreatureFullyConnected8(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = hyperparameters["dimensions"]

        self.model = torch.nn.Sequential(
            torch.nn.Linear(12, 1024),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm1d(1024),
            torch.nn.Linear(1024, 7*7*128),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm1d(6272),
            torch.nn.ReLU(),
            torch.nn.Linear(6272, 64),
            torch.nn.LeakyReLU(0.05),
            #torch.nn.BatchNorm1d(64),
            torch.nn.Linear(64, 4)


        )
        self.optimizer = torch.optim.Adam(self.model.parameters())

class CreatureFullyConnected9(CreatureNetwork):
    def __init__(self, hyperparameters):
        super().__init__(hyperparameters)
        dims = hyperparameters["dimensions"]

        # [dims are as follows: -1,30,30,30,30,30,30,30,30,30,30,10,10,10,10,10,10,10,10,10,10,30,30,30,30,30,30,30,30,30,30,4]
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
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dims[5], dims[6]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[6], dims[7]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[7], dims[8]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[8], dims[9]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[9], dims[10]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[10], dims[11]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[11], dims[12]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[12], dims[13]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[13], dims[14]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[14], dims[15]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[15], dims[16]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[16], dims[17]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[17], dims[18]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[18], dims[19]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[19], dims[20]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[20], dims[21]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(dims[21], dims[22]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[22], dims[23]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[23], dims[24]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[24], dims[25]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[25], dims[26]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[26], dims[27]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[27], dims[28]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[28], dims[29]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[29], dims[30]),
            torch.nn.ReLU(),
            torch.nn.Linear(dims[30], dims[31])
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())


        
