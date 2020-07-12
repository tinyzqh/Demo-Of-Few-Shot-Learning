import math
import random
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
def generate_dataset(number_of_examples, test=False):
    if test:
        b = math.pi
    else:
        b = 0 if random.choice([True, False]) else math.pi # setting up beta variable randomly
    x = torch.rand(number_of_examples, 1)*4*math.pi - 2*math.pi # randomly generate x datapoints
    y = torch.sin(x + b) # generate labels in form of sine-curve for randomly generated x
    return x,y
def NeuralNetwork(x, params):
    x = F.linear(x, params[0], params[1])
    x = F.relu(x)
    x = F.linear(x, params[2], params[3])
    x = F.relu(x)
    x = F.linear(x, params[4], params[5])
    return x

# define Initial Parameters
theta = [
    torch.Tensor(32, 1).uniform_(-1., 1.).requires_grad_(),
    torch.Tensor(32).zero_().requires_grad_(),

    torch.Tensor(32, 32).uniform_(-1./math.sqrt(32), 1./math.sqrt(32)).requires_grad_(),
    torch.Tensor(32).zero_().requires_grad_(),

    torch.Tensor(1, 32).uniform_(-1./math.sqrt(32), 1./math.sqrt(32)).requires_grad_(),
    torch.Tensor(1).zero_().requires_grad_(),
]


alpha, beta = 3e-2, 1e-2 # hyper-paramaters as shown in algorithm above.
opt = torch.optim.SGD(theta, lr=beta) # Optimizer
number_of_examples= 5 # 5-shot learning, 5 examples per task.
iterations = 100000 # number of iterations to enable the model to reach optimal point
epochs = 4 # number of iterations to train task specific model

for it in range(iterations):  # training for 1 million iterations
    b = 0 if random.choice([True, False]) else math.pi  # setting up beta variable randomly

    #### Randomly obtain task 1 sinusoidal data ####
    x = torch.rand(4, 1) * 4 * math.pi - 2 * math.pi
    y = torch.sin(x + b)

    #### Randomly obtain the task 2 sinusoidal data ####
    v_x = torch.rand(4, 1) * 4 * math.pi - 2 * math.pi
    v_y = torch.sin(v_x + b)

    opt.zero_grad()  # setup optimizer

    new_params = theta  # initialize weights for inner loop
    for k in range(4):
        f = NeuralNetwork(x, new_params)  # re-initialize task 2 neural network with new parameters
        loss = F.l1_loss(f, y)  # set loss as L1 Loss

        # create_graph=True because computing grads here is part of the forward pass.
        # We want to differentiate through the SGD update steps and get higher order
        # derivatives in the backward pass.
        grads = torch.autograd.grad(loss, new_params, create_graph=True)
        new_params = [(new_params[i] - alpha * grads[i]) for i in range(len(theta))]  # update weights of inner loop

    v_f = NeuralNetwork(v_x, new_params)  # re-initialize task 1 neural network with new parameters
    loss2 = F.l1_loss(v_f, v_y)  # calculate Loss
    loss2.backward()  # Backward Pass

    opt.step()

    if it % 1000 == 0:
        print('Iteration %d Loss %.4f' % (it, loss2))

# Randomly generate 5 data points.
t_b = math.pi
t_x = torch.rand(4, 1)*4*math.pi - 2*math.pi
t_y = torch.sin(t_x + t_b)

opt.zero_grad()
n_inner_loop = 5
t_params = theta
for k in range(n_inner_loop):
    t_f = NeuralNetwork(t_x, t_params)
    t_loss = F.l1_loss(t_f, t_y)

    grads = torch.autograd.grad(t_loss, t_params, create_graph=True)
    t_params = [(t_params[i] - alpha*grads[i]) for i in range(len(theta))]

test_x = torch.arange(-2*math.pi, 2*math.pi, step=0.01).unsqueeze(1)
test_y = torch.sin(test_x + t_b)

test_f = NeuralNetwork(test_x, t_params)

plt.plot(test_x.data.numpy(), test_y.data.numpy(), label='sin(x)')
plt.plot(test_x.data.numpy(), test_f.data.numpy(), label='net(x)')
plt.plot(t_x.data.numpy(), t_y.data.numpy(), 'o', label='Examples')
plt.show()