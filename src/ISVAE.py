import torch
from src.VAE import *
from src.filter_bank import *


class ISVAE(nn.Module):
    def __init__(self, x_dim, z_dim, number_f, filter_w, psd, version):
        super(ISVAE, self).__init__()

        self.number_f = number_f
        self.filter_w = filter_w
        self.x_dim = x_dim
        self.z_dim = z_dim  
        self.psd = psd
        self.version = version

        self.filter_bank = FB(number_f, x_dim, filter_w)
        self.vae = VAE(x_dim, z_dim, number_f, filter_w, psd, version)


    def forward(self, x):
        H_list, x_filtered_list, x_filtered_sum_list, f_0_list = self.filter_bank(x)

        f_0_torch = torch.cat(tuple(f_0_list),1)
        f_axis = torch.arange(x_filtered_list[0].shape[1])
        f_axis = f_axis.unsqueeze(0).repeat(x_filtered_list[0].shape[0], 1)

        aux = 0

        for f in range(self.number_f):
            if len(list(f_0_list[0].shape)) == 1:
                f_0_list[f] = f_0_list[f].unsqueeze(1)
            aux = aux + x* torch.exp(-0.5*((f_axis-f_0_list[f].repeat((1, x_filtered_list[0].shape[1])))**2/self.filter_w))
        new_x =aux

        if self.version == 2:
            z_mean, z_sigma, z, x_rec, f_0_rec = self.vae(f_0_torch,x)
            return new_x, x_rec, z_mean, z_sigma, z, H_list, x_filtered_list, x_filtered_sum_list, f_0_list, f_0_rec
        else:
            z_mean, z_sigma, z, x_rec = self.vae(f_0_torch, x)
            return new_x, x_rec, z_mean, z_sigma, z, H_list, x_filtered_list, x_filtered_sum_list, f_0_list



        