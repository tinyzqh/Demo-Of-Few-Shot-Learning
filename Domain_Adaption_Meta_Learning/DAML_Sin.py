import math
import random
import sys
import torch # v0.4.1
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from time import sleep
import matplotlib as mpl
import matplotlib.pyplot as plt

def meta_net(x, params):
    # main network which is suppose to learn our main objective i.e; learn sinusoidal curve family here.
    x = F.linear(x, params[0], params[1])
    x1 = F.relu(x)

    x = F.linear(x1, params[2], params[3])
    x2 = F.relu(x)

    y = F.linear(x2, params[4], params[5])

    return y, x2, x1

params = [
    torch.Tensor(32, 1).uniform_(-1., 1.).requires_grad_(),
    torch.Tensor(32).zero_().requires_grad_(),

    torch.Tensor(32, 32).uniform_(-1./math.sqrt(32), 1./math.sqrt(32)).requires_grad_(),
    torch.Tensor(32).zero_().requires_grad_(),

    torch.Tensor(1, 32).uniform_(-1./math.sqrt(32), 1./math.sqrt(32)).requires_grad_(),
    torch.Tensor(1).zero_().requires_grad_(),
]


def adap_net(y, x2, x1, params):
    # the net takes forward pass from meta_net and provides efficient parameter initializations i.e;
    # It works adapts the meta_net easily to any form of change

    x = torch.cat([y, x2, x1], dim=1)

    x = F.linear(x, params[0], params[1])
    x = F.relu(x)

    x = F.linear(x, params[2], params[3])
    x = F.relu(x)

    x = F.linear(x, params[4], params[5])

    return x


adap_params = [
    torch.Tensor(32, 1 + 32 + 32).uniform_(-1. / math.sqrt(65), 1. / math.sqrt(65)).requires_grad_(),
    torch.Tensor(32).zero_().requires_grad_(),

    torch.Tensor(32, 32).uniform_(-1. / math.sqrt(32), 1. / math.sqrt(32)).requires_grad_(),
    torch.Tensor(32).zero_().requires_grad_(),

    torch.Tensor(1, 32).uniform_(-1. / math.sqrt(32), 1. / math.sqrt(32)).requires_grad_(),
    torch.Tensor(1).zero_().requires_grad_(),
]

opt = torch.optim.SGD(params + adap_params, lr=1e-2)
n_inner_loop = 5
alpha = 3e-2

inner_loop_loss = []
outer_lopp_loss = []

# Here, T ? p(T ) {or minibatch of tasks} is to learn sinusoidal family curves

with tqdm(total=100000, file=sys.stdout) as pbar:
    for it in range(100000):
        b = 0 if random.choice([True, False]) else math.pi
        #### Randomly obtain the task 2 sinusoidal data ####

        # Sample robotic task data d_r~D_r
        v_x = torch.rand(4, 1) * 4 * math.pi - 2 * math.pi
        v_y = torch.sin(v_x + b)

        opt.zero_grad()

        new_params = params
        for k in range(n_inner_loop):
            # Sample Human task data d_h~D_h
            sampled_data = torch.FloatTensor([[random.uniform(math.pi / 4, math.pi / 2)
                                               if b == 0 else random.uniform(-math.pi / 2, -math.pi / 4)]])

            # Here, si is adap_net parameters: adap_params and theta is meta_net parameters: new_params
            f, f2, f1 = meta_net(sampled_data, new_params)
            h = adap_net(f, f2, f1, adap_params)

            # calculate loss
            adap_loss = F.l1_loss(h, torch.zeros(1, 1))
            grads = torch.autograd.grad(adap_loss, new_params, create_graph=True)

            # Compute policy parameters phi_t(new_params)
            new_params = [(new_params[i] - alpha * grads[i]) for i in range(len(params))]

            if it % 100 == 0:
                inner_loop_loss.append(adap_loss)

        v_f, _, _ = meta_net(v_x, new_params)  # forward pass using learned policy parameters phi_t
        loss = F.l1_loss(v_f, v_y)  # calculate the loss of meta_net
        loss.backward()

        opt.step()  # optimize the policy parameters(theta and si)
        pbar.update(1)
        if it % 100 == 0:
            outer_lopp_loss.append(loss)
            # print ('Iteration %d -- Outer Loss: %.4f' % (it, loss))

t_b = math.pi
opt.zero_grad()
t_params = params

for k in range(n_inner_loop):
    # sample the new task data
    new_task_data = torch.FloatTensor([[random.uniform(math.pi/4, math.pi/2)
                                        if t_b == 0 else random.uniform(-math.pi/2, -math.pi/4)]])
    # forward pass through meta_net to extract the input for adap_net
    t_f, t_f2, t_f1 = meta_net(new_task_data, t_params)
    # extract the information from adap_net
    t_h = adap_net(t_f, t_f2, t_f1, adap_params)
    # calculate the loss, here we used true label as torch.zeros(1, 1), because t_b = pi
    t_adap_loss = F.l1_loss(t_h, torch.zeros(1, 1))

    grads = torch.autograd.grad(t_adap_loss, t_params, create_graph=True)
    # learn the policy using the loss of adap_net
    t_params = [(t_params[i] - alpha*grads[i]) for i in range(len(params))]


test_x = torch.arange(-2*math.pi, 2*math.pi, step=0.01).unsqueeze(1)
test_y = torch.sin(test_x + t_b)

test_f, _, _ = meta_net(test_x, t_params) # use the learned parameters

plt.plot(test_x.data.numpy(), test_y.data.numpy(), label='sin(x)')
plt.plot(test_x.data.numpy(), test_f.data.numpy(), label='meta_net(x)')
# plt.legend()
# plt.savefig('daml-sine.png')
plt.show()

def plot_loss(inner_loop_loss,name="Loss Curve"):
    plt.plot(inner_loop_loss, label=name)
    plt.show()
plot_loss(outer_lopp_loss)