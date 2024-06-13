
# The N-ISAC paper is implemented using 'snn' package in https://github.com/kclip/snn, which has not been updated for some time and is somewhat challenging to use.
# This implementation is based on the 'snntorch' https://snntorch.readthedocs.io/en/latest/, which is easier to implement and see how N-ISAC works.
# It offers various hyperparameters to experiment with, which may yield better results and serve as a starting point for further extensions.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
import matplotlib.pyplot as plt
from snntorch import surrogate

dtype = torch.float
num_bw_expansion = 5  

# Network Architecture (to be optimized)
num_inputs = 2 * num_bw_expansion * 2
num_hidden = 10 
num_outputs = 2

# static (0) or varying (1) data; TMode = 0 is valid only when varying = 0
varying_decoding = 1  # if 0, TMode can be 0 or 1; if 1, TMode must be only 1.
varying_target = 0


TMode = 1  # Mode 0 is valid only when the time samples across all time steps for each example are the same / static

num_epochs = 10
control = 0.3 
channel_fix = 0 
num_steps = 80
total_batch_train = 468
total_batch_test = 78
batch_size = 128  
num_transmit = 1  

# initialize training data
if varying_decoding:
    torch.manual_seed(1)
    data_train = torch.bernoulli(torch.rand(total_batch_train, num_steps, batch_size, num_transmit))
    torch.manual_seed(1)
    data_test = torch.bernoulli(torch.rand(total_batch_test, num_steps, batch_size, num_transmit))
else:
    torch.manual_seed(1)
    data_train = torch.bernoulli(torch.rand(total_batch_train, 1, batch_size, num_transmit))
    data_train = data_train.repeat(1, num_steps, 1, 1)
    torch.manual_seed(1)
    data_test = torch.bernoulli(torch.rand(total_batch_test, 1, batch_size, num_transmit))
    data_test = data_test.repeat(1, num_steps, 1, 1)

# initialize the presence of the target
Mode = 1
if Mode == 0:  
    target_train = data_train
    target_test = data_test
elif Mode == 1:  # the target is randomly present/absent
    if varying_target:
        torch.manual_seed(2)
        target_train = torch.bernoulli(torch.rand(total_batch_train, num_steps, batch_size, num_transmit))
        torch.manual_seed(2)
        target_test = torch.bernoulli(torch.rand(total_batch_test, num_steps, batch_size, num_transmit))
    else:
        torch.manual_seed(2)
        target_train = torch.bernoulli(torch.rand(total_batch_train, 1, batch_size, num_transmit))
        target_train = target_train.repeat(1, num_steps, 1, 1)
        torch.manual_seed(2)
        target_test = torch.bernoulli(torch.rand(total_batch_test, 1, batch_size, num_transmit))
        target_test = target_test.repeat(1, num_steps, 1, 1)
elif Mode == 2:  # the target is absent all the time
    target_train = torch.zeros(total_batch_train, num_steps, batch_size, num_transmit)
    target_test = torch.zeros(total_batch_test, num_steps, batch_size, num_transmit)
elif Mode == 3:  # the target is present all the time
    target_train = torch.ones(total_batch_train, num_steps, batch_size, num_transmit)
    target_test = torch.ones(total_batch_test, num_steps, batch_size, num_transmit)


# modulation of dataset
data_modulated_train = torch.zeros(total_batch_train, num_steps, batch_size, num_transmit * 2 * num_bw_expansion)  # PPM modulation and bandwidth expansion
data_modulated_train[:, :, :, 0] = 1 - data_train[:, :, :, 0] 
data_modulated_train[:, :, :, num_bw_expansion] = data_train[:, :, :, 0]  
data_modulated_train = data_modulated_train + 1j * 0  

shape_train = torch.Tensor.size(data_modulated_train) 
target_training = torch.zeros(shape_train)
for j in range(num_transmit * 2 * num_bw_expansion):
    target_training[:, :, :, j] = target_train[:, :, :, 0]

data_modulated_test = torch.zeros(total_batch_test, num_steps, batch_size, num_transmit * 2 * num_bw_expansion)  # PPM modulation
data_modulated_test[:, :, :, 0] = 1 - data_test[:, :, :, 0]
data_modulated_test[:, :, :, num_bw_expansion] = data_test[:, :, :, 0]
data_modulated_test = data_modulated_test + 1j * 0

shape_test = torch.Tensor.size(data_modulated_test)
target_testing = torch.zeros(shape_test)
for j in range(num_transmit * 2 * num_bw_expansion):
    target_testing[:, :, :, j] = target_test[:, :, :, 0]

device = torch.device("cpu")

# Temporal Dynamics
spike_grad = surrogate.fast_sigmoid(slope=25)
beta = 0.5

# Define the snn at the receiver
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):  # x would be of size [80, 128, 4]

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

receiver = Net().to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(receiver.parameters(), lr=5e-4, betas=(0.9, 0.999))


data_modulated_train = data_modulated_train.transpose(1, 2)
shape_noise_train = torch.Tensor.size(data_modulated_train)

data_distorted_train1 = torch.roll(data_modulated_train, 1)  # the delay is one time step; feel free to change the delay
data_distorted_train1[:, :, 0, 0] = 0  
# The second delayed signal
data_distorted_train2 = torch.roll(data_modulated_train, 2)
data_distorted_train2[:, :, 0, 0] = 0
data_distorted_train2[:, :, 0, 1] = 0
# The third delayed signal
data_distorted_train3 = torch.roll(data_modulated_train, 3)
data_distorted_train3[:, :, 0, 0] = 0
data_distorted_train3[:, :, 0, 1] = 0
if torch.Tensor.size(data_modulated_train)[3] > 2:
    data_distorted_train3[:, :, 0, 2] = 0
else:
    data_distorted_train3[:, :, 1, 0] = 0
# The fourth delayed signal
data_distorted_train4 = torch.roll(data_modulated_train, 4)
data_distorted_train4[:, :, 0, 0] = 0
data_distorted_train4[:, :, 0, 1] = 0
if torch.Tensor.size(data_modulated_train)[3] > 2:
    data_distorted_train4[:, :, 0, 2] = 0
    data_distorted_train4[:, :, 0, 3] = 0
else:
    data_distorted_train4[:, :, 1, 0] = 0
    data_distorted_train4[:, :, 1, 1] = 0


data_modulated_test = data_modulated_test.transpose(1, 2)  
shape_noise_test = torch.Tensor.size(data_modulated_test)

# The first delayed signal
data_distorted_test1 = torch.roll(data_modulated_test, 1)
data_distorted_test1[:, :, 0, 0] = 0
# The second delayed signal
data_distorted_test2 = torch.roll(data_modulated_test, 2)
data_distorted_test2[:, :, 0, 0] = 0
data_distorted_test2[:, :, 0, 1] = 0
# The third delayed signal
data_distorted_test3 = torch.roll(data_modulated_test, 3)
data_distorted_test3[:, :, 0, 0] = 0
data_distorted_test3[:, :, 0, 1] = 0
if torch.Tensor.size(data_modulated_test)[3] > 2:
    data_distorted_test3[:, :, 0, 2] = 0
else:
    data_distorted_test3[:, :, 1, 0] = 0
# The fourth delayed signal
data_distorted_test4 = torch.roll(data_modulated_test, 4)
data_distorted_test4[:, :, 0, 0] = 0
data_distorted_test4[:, :, 0, 1] = 0
if torch.Tensor.size(data_modulated_test)[3] > 2:
    data_distorted_test4[:, :, 0, 2] = 0
    data_distorted_test4[:, :, 0, 3] = 0
else:
    data_distorted_test4[:, :, 1, 0] = 0
    data_distorted_test4[:, :, 1, 1] = 0

# Generate channel
if channel_fix:
    h_target_train = (0.8 * torch.ones(shape_train) + 1j * 0.8 * torch.ones(shape_train)) * target_training
    h_clutter_train1 = 0.8 * torch.ones(shape_train) + 1j * 0.8 * torch.ones(shape_train)
    h_clutter_train2 = 0.8 * torch.ones(shape_train) + 1j * 0.8 * torch.ones(shape_train)
    h_clutter_train3 = 0.8 * torch.ones(shape_train) + 1j * 0.8 * torch.ones(shape_train)
    h_clutter_train4 = 0.8 * torch.ones(shape_train) + 1j * 0.8 * torch.ones(shape_train)
else:
    h_target_train = (torch.normal(mean=0.4, std=np.sqrt(1 / 2), size=shape_train) + 1j * torch.normal(mean=0.4,
                                                                                                     std=np.sqrt(
                                                                                                         1 / 2),
                                                                                                     size=shape_train)) * target_training
    h_clutter_train1 = torch.normal(mean=0.4, std=np.sqrt(1 / 2), size=shape_train) + 1j * torch.normal(mean=0.4,
                                                                                                     std=np.sqrt(
                                                                                                         1 / 2),
                                                                                                     size=shape_train)
    h_clutter_train2 = torch.normal(mean=0.4, std=np.sqrt(1 / 2), size=shape_train) + 1j * torch.normal(mean=0.4,
                                                                                                     std=np.sqrt(
                                                                                                         1 / 2),
                                                                                                     size=shape_train)
    h_clutter_train3 = torch.normal(mean=0.4, std=np.sqrt(1 / 2), size=shape_train) + 1j * torch.normal(mean=0.4,
                                                                                                     std=np.sqrt(
                                                                                                         1 / 2),
                                                                                                     size=shape_train)
    h_clutter_train4 = torch.normal(mean=0.4, std=np.sqrt(1 / 2), size=shape_train) + 1j * torch.normal(mean=0.4,
                                                                                                     std=np.sqrt(
                                                                                                         1 / 2),
                                                                                                     size=shape_train)

if channel_fix:
    h_target_test = (0.8 * torch.ones(shape_test) + 1j * 0.8 * torch.ones(shape_test)) * target_testing
    h_clutter_test1 = 0.8 * torch.ones(shape_test) + 1j * 0.8 * torch.ones(shape_test)
    h_clutter_test2 = 0.8 * torch.ones(shape_test) + 1j * 0.8 * torch.ones(shape_test)
    h_clutter_test3 = 0.8 * torch.ones(shape_test) + 1j * 0.8 * torch.ones(shape_test)
    h_clutter_test4 = 0.8 * torch.ones(shape_test) + 1j * 0.8 * torch.ones(shape_test)
else:
    h_target_test = (torch.normal(mean=0.4, std=np.sqrt(1 / 2), size=shape_test) + 1j * torch.normal(mean=0.4,
                                                                                                   std=np.sqrt(
                                                                                                       1 / 2),
                                                                                                   size=shape_test)) * target_testing
    h_clutter_test1 = torch.normal(mean=0.4, std=np.sqrt(1 / 2), size=shape_test) + 1j * torch.normal(mean=0.4,
                                                                                                   std=np.sqrt(
                                                                                                       1 / 2),
                                                                                                   size=shape_test)
    h_clutter_test2 = torch.normal(mean=0.4, std=np.sqrt(1 / 2), size=shape_test) + 1j * torch.normal(mean=0.4,
                                                                                                   std=np.sqrt(
                                                                                                       1 / 2),
                                                                                                   size=shape_test)
    h_clutter_test3 = torch.normal(mean=0.4, std=np.sqrt(1 / 2), size=shape_test) + 1j * torch.normal(mean=0.4,
                                                                                                   std=np.sqrt(
                                                                                                       1 / 2),
                                                                                                   size=shape_test)
    h_clutter_test4 = torch.normal(mean=0.4, std=np.sqrt(1 / 2), size=shape_test) + 1j * torch.normal(mean=0.4,
                                                                                                   std=np.sqrt(
                                                                                                       1 / 2),
                                                                                                   size=shape_test)


h_target_train = h_target_train.transpose(1, 2) 
h_clutter_train1 = h_clutter_train1.transpose(1, 2)
h_clutter_train2 = h_clutter_train2.transpose(1, 2)
h_clutter_train3 = h_clutter_train3.transpose(1, 2)
h_clutter_train4 = h_clutter_train4.transpose(1, 2)

input_complex_train = data_distorted_train1 * h_clutter_train1 + data_distorted_train2 * h_clutter_train2 + data_distorted_train3 * h_clutter_train3 + data_distorted_train4 * h_clutter_train4 + data_modulated_train * h_target_train + torch.normal(mean=0, std=np.sqrt(1 / 10), size=shape_noise_train)
input_complex_train = input_complex_train.transpose(1, 2)
input_train = torch.cat((torch.real(input_complex_train), torch.imag(input_complex_train)), 3) 
input_train = torch.square(input_train)

h_target_test = h_target_test.transpose(1, 2)
h_clutter_test1 = h_clutter_test1.transpose(1, 2)
h_clutter_test2 = h_clutter_test2.transpose(1, 2)
h_clutter_test3 = h_clutter_test3.transpose(1, 2)
h_clutter_test4 = h_clutter_test4.transpose(1, 2)

input_complex_test = data_distorted_test1 * h_clutter_test1 + data_distorted_test2 * h_clutter_test2 + data_distorted_test3 * h_clutter_test3 + data_distorted_test4 * h_clutter_test4 + data_modulated_test * h_target_test + torch.normal(mean=0, std=np.sqrt(1 / 10), size=shape_noise_test)
input_complex_test = input_complex_test.transpose(1, 2)
input_test = torch.cat((torch.real(input_complex_test), torch.imag(input_complex_test)), 3) 
input_test = torch.square(input_test)

# Training 
for epoch in range(num_epochs):
    for i in range(total_batch_train):
        data = input_train[i].to(device)  
        label_decoding = data_train[i, :, :, 0].to(device)  
        label_sensing = target_train[i, :, :, 0].to(device)  

        # forward pass
        receiver.train()
        spk_rec, mem_rec = receiver(data)  

        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val = loss_val + control * loss(mem_rec[step, :, 0], label_decoding[step]) + (1 - control) * loss(
                mem_rec[step, :, 1], label_sensing[step])

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    # Testing
    if TMode == 0:
        total = 0
        correct_decoding = 0
        correct_sensing = 0
        with torch.no_grad():
            receiver.eval()
            for i in range(total_batch_test):
                data_testing = input_test[i].to(device)  # [80, 128, 4 * num_bw_expansion]
                label_decoding = data_test[i, :, :, 0].to(device)  # [80, 128]
                label_sensing = target_test[i, :, :, 0].to(device)  # [80, 128]

                # forward pass
                test_spk, test_mem = receiver(data_testing)  # [80, 128, 2]

                total += label_decoding.size(1)
                correct_decoding += (((test_spk.sum(dim=0)[:, 0] > 40) + 0) == label_decoding[0]).sum().item()
                correct_sensing += (((test_spk.sum(dim=0)[:, 1] > 40) + 0) == label_sensing[0]).sum().item()

            print(f"Test Accuracy for Decoding in epoch {epoch}: {100 * correct_decoding / total:.2f}%")
            print(f"Test Accuracy for Sensing in epoch {epoch}: {100 * correct_sensing / total:.2f}%")
    elif TMode == 1:
        correct_decoding = 0
        correct_sensing = 0
        with torch.no_grad():
            receiver.eval()
            total = total_batch_test 
            total_s = 0

            for i in range(total_batch_test):
                data_testing = input_test[i].to(device) 
                label_decoding = data_test[i, :, :, 0].to(device) 
                label_sensing = target_test[i, :, :, 0].to(device) 

                # forward pass
                test_spk, test_mem = receiver(data_testing) 

                correct_decoding += torch.sum(
                    torch.sum(((test_spk[:, :, 0] == label_decoding) + 0), 0) / num_steps) / label_decoding.size(
                    1) 

                correct_sensing += (((test_spk.sum(dim=0)[:, 1] > 40) + 0) == label_sensing[0]).sum().item()
                total_s = total_s + 128

            print(f"Test Accuracy for Decoding in epoch {epoch}: {100 * correct_decoding / total:.2f}%")
            print(f"Test Accuracy for Sensing in epoch {epoch}: {100 * correct_sensing / total_s:.2f}%")
