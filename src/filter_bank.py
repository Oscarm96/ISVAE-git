import torch
from torch import nn
from src.util import *



class FB(nn.Module):
    def __init__(self, number_f, D_in,  filter_w):
        super().__init__()

        self.number_f = number_f  # Number of filters
        self.x_dim = D_in  # Input signal dimension
        self.filter_w = filter_w  # Filter width

        self.keys_list = [str(x) for x in range(number_f)] # Dictionary keys
        values = [None for _ in range(number_f)] # Dictionary values at init (None)

        self.dic_FB = torch.nn.ModuleDict(zip(self.keys_list, values)) # Dictionary of modules
        self.dic_LN = torch.nn.ModuleDict(zip(self.keys_list, values))

        self.build_banks()

    def build_banks(self):
        # Setup according to dataset type
        for i in self.keys_list:
            self.build_bank(i, [1,3],[3,1], [2,2], [36,20,10,1])

    def build_bank(self, i, channels_list, conv_outputs, conv_strides, linear_inputs):
        # SA bank: alternating Conv1d and MaxPool1d layers
        SA_bank = torch.nn.ModuleList()
        for out, stride, ch in zip(conv_outputs, conv_strides, channels_list):
            SA_bank.append(torch.nn.Conv1d(ch, out, 3, stride))
            SA_bank.append(torch.nn.MaxPool1d(3, 2))
            SA_bank.append(torch.nn.ReLU())

        # LN bank: series of Linear layers
        LN_bank = torch.nn.ModuleList()
        if len(linear_inputs) == 3:
            LN_bank.append(torch.nn.Linear(linear_inputs[0], linear_inputs[1]))
            LN_bank.append(torch.nn.ReLU()) 
            LN_bank.append(torch.nn.Linear(linear_inputs[1], linear_inputs[2]))
            LN_bank.append(torch.nn.Sigmoid())
        else:
            LN_bank.append(torch.nn.Linear(linear_inputs[0], linear_inputs[1]))
            LN_bank.append(torch.nn.ReLU()) 
            LN_bank.append(torch.nn.Linear(linear_inputs[1], linear_inputs[2]))
            LN_bank.append(torch.nn.ReLU()) 
            LN_bank.append(torch.nn.Linear(linear_inputs[2], linear_inputs[3]))
            LN_bank.append(torch.nn.Sigmoid())

        self.dic_FB[i] = SA_bank
        self.dic_LN[i] = LN_bank

    def forward(self, x):
        H_list, x_filtered_list, x_filtered_sum_list, f_0_list = [], [], [], []

        for i in self.keys_list:
            x_aux = x - sum(x_filtered_list) if i != '0' else x
            x_filtered_sum_list.append(x_aux)

            for layer in self.dic_FB[i]:
                x_aux = x_aux.unsqueeze(1) if layer == self.dic_FB[i][0] else x_aux
                x_aux = layer(x_aux)
            for layer in self.dic_LN[i]:
                x_aux = x_aux.squeeze(1) if layer == self.dic_LN[i][0] else x_aux
                x_aux = layer(x_aux)


            f_0_batch = self.x_dim * x_aux
            f = torch.arange(self.x_dim)
            H_aux = torch.zeros_like(x)
           
            for j in range(H_aux.shape[0]):
                H_aux[j, :] = normal(f, f_0_batch[j], self.filter_w)

            H_list.append(H_aux)
            x_filtered_list.append(x_filtered_sum_list[-1] * H_list[-1])
            f_0_list.append(f_0_batch)

        return H_list, x_filtered_list, x_filtered_sum_list, f_0_list



