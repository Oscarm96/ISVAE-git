import torch
from torch import nn
from src.util import *



class Encoder(torch.nn.Module):
    def __init__(self, number_f): 
        super(Encoder, self).__init__()

        self.number_f = number_f
        module_list = torch.nn.ModuleList([
                        torch.nn.Linear(self.number_f, self.number_f),
                        nn.ReLU(),
                        torch.nn.Linear(self.number_f, self.number_f),
                        nn.ReLU(),
                        torch.nn.Linear(self.number_f, self.number_f),
                    ])
        self.layers = module_list

    def forward(self, x):
        for layer in self.layers:
                x = layer(x)
        return x
    
class Decoder_v1(nn.Module): # Vanila
    def __init__(self, z_dim, D_in):
        super().__init__()

        layer_conf = [(z_dim, 70), (70, 150), (150, 300), (300, D_in)]
        self.layers = self.build_layers(layer_conf)

    @staticmethod
    def build_layers(layer_config):
        layers = nn.ModuleList()
        for in_dim, out_dim in layer_config:
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers[-1] = nn.Linear(layer_config[-1][1], layer_config[-1][1])

        return layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    
class Decoder_v2(torch.nn.Module): #ISVAE decoder, añadiendo z
    def __init__(self, z_dim, D_in, number_f, filter_w, psd):
        super().__init__()

        self.number_f = number_f
        self.D_in = D_in
        self.filter_w = filter_w
        self.psd = psd
        self.z_dim = z_dim

        layer_conf_1 = [(D_in, 300), (300, 150), (150, 70), (70, self.number_f)]
        layer_conf_2 = [(self.number_f+self.z_dim, 70), (70, 150), (150, 300), (300, D_in)]


        self.pre = torch.nn.ModuleList([
            torch.nn.Linear(z_dim, self.number_f),
            nn.ReLU(),
            torch.nn.Linear(self.number_f, self.number_f),
            nn.ReLU(),
            torch.nn.Linear(self.number_f, self.number_f),
            nn.Sigmoid(),

        ])
        
        self.generator_1 = self.build_layers(layer_conf_1)
        self.generator_2 = self.build_layers(layer_conf_2)
    

    @staticmethod
    def build_layers(layer_config):
        layers = nn.ModuleList()
        for in_dim, out_dim in layer_config:
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU()])
        layers[-1] = nn.Linear(layer_config[-1][1], layer_config[-1][1])

        return layers
    
    def forward(self,f_0, z, x_orig):

        f_0_aux = z
        for layer in self.pre:
            f_0_aux = layer(f_0_aux)

        f_0_rec = self.D_in * f_0_aux 
        f = torch.arange(self.D_in)
        H_f = torch.zeros_like(x_orig)

        for j in range(H_f.shape[0]):
            H_aux = 0
            for i in range(self.number_f):
                H_aux = H_aux + normal(f, f_0_rec[j, i], self.filter_w)
            H_f[j, :] = H_aux

        
        
        x_rec = self.psd.repeat((x_orig.shape[0], 1)) * H_f 

        x_rec_gen_1 = x_rec 
        for layer in self.generator_1:
            x_rec_gen_1 = layer(x_rec_gen_1)

    
        x_rec_gen_1 = torch.cat((x_rec_gen_1, z), dim=1)
        x_rec_final = x_rec_gen_1 
        for layer in self.generator_2:
            x_rec_final = layer(x_rec_final)


        return x_rec_final, f_0_rec
    

class VAE(nn.Module):
    def __init__(self, x_dim, z_dim, number_f, filter_w, psd, version=1):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        self.number_f = number_f
        self.encoder = Encoder(number_f)
        self.version = version
        
        if version == 1:
            self.decoderv1 = Decoder_v1(z_dim, x_dim)
        elif version == 2:
            self.decoderv2 = Decoder_v2(z_dim, x_dim, number_f, filter_w, psd)
        else:
            raise ValueError('Wrong version value')

        self._enc_aux = torch.nn.Linear(self.number_f, self.number_f)
        self._enc_mu = torch.nn.Linear(self.number_f, z_dim)
        self._enc_var = torch.nn.Linear(self.number_f, z_dim)
    
    def _sample_latent(self, h_enc):
        aux =torch.nn.functional.relu(self._enc_aux(h_enc))
        mu = self._enc_mu(aux)
        var = torch.nn.functional.softplus(self._enc_var(aux)) + 1e-20
        eps = torch.randn_like(var)
        return mu, var, mu + torch.sqrt(var) * eps
    
    @staticmethod
    def latent_loss(z_mean, z_stddev):
        kl = 0.5 * torch.mean(torch.sum(z_stddev, dim=1) + torch.sum(z_mean ** 2, dim=1) - z_mean.shape[-1] - torch.sum(torch.log(z_stddev), dim=1))
        return kl

    @staticmethod
    def reconstruction_VAE(x, x_rec):
        criterion = torch.nn.MSELoss()
        return criterion(x, x_rec)

    def forward(self, f_0, x_orig):
        h_enc = self.encoder(f_0)
        
        z_mean, z_var, z = self._sample_latent(h_enc)
        if self.version == 1:
            x_rec = self.decoderv1(z)
            return z_mean, z_var, z, x_rec
        else:
            x_rec, f_0_rec = self.decoderv2(f_0,z, x_orig)
            return z_mean, z_var, z, x_rec, f_0_rec
        


       


