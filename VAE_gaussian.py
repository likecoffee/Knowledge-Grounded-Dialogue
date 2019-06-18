import torch
import numpy as np
from torch import nn
from torch.nn import functional

def gaussian_kld(mu_1, logvar_1, mu_2, logvar_2, mean=False):
    loss = (logvar_2 - logvar_1) + (torch.exp(logvar_1) / torch.exp(logvar_2)) + ((mu_1 - mu_2) ** 2 / torch.exp(logvar_2) - 1.)
    loss = loss / 2
    if mean:
        loss = torch.mean(loss)
    else:
        loss = torch.sum(loss)
    #stochastic_list = [mu_1.detach().cpu().numpy(), logvar_1.detach().cpu().numpy(), mu_2.detach().cpu().numpy(), logvar_2.detach().cpu().numpy()]
    #return loss, stochastic_list
    return loss

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_size, addtional_input_size, mlp_size, z_size):
        self.input_size = input_size
        self.mlp_size = mlp_size
        self.additional_input_size = addtional_input_size
        self.z_size = z_size
        super(VariationalAutoEncoder, self).__init__()
        self.inference_linear = nn.Sequential(
            nn.Linear(input_size + addtional_input_size, mlp_size),
            nn.LeakyReLU(),
            nn.Linear(mlp_size, 2 * z_size, bias=False)
        )
        self.prior_linear = nn.Sequential(
            nn.Linear(input_size, mlp_size),
            nn.LeakyReLU(),
            nn.Linear(mlp_size, 2 * z_size, bias=False)
        )

    @staticmethod
    def reparameter(mu, logvar, random_variable=None):
        std = logvar.mul(0.5).exp_()
        if random_variable is None:
            random_variable = mu.new(*mu.size()).normal_()
            return random_variable.mul(std).add_(mu)
        else:
            if len(random_variable.size()) == 3:
                sampled_random_variable = random_variable.mul(std.unsqueeze(0)).add_(mu.unsqueeze(0))
                return random_variable
            elif len(random_variable.size()) == 2:
                return random_variable.mul(std).add_(mu)
            else:
                raise Exception("Wrong size of given random variable")

    def forward(self, input, additional_input=None, random_variable=None, inference_mode=True):
        prior_gaussian_paramter = self.prior_linear(input)
        prior_gaussian_paramter = torch.clamp(prior_gaussian_paramter, -4, 4)
        prior_mu, prior_logvar = torch.chunk(prior_gaussian_paramter, 2, 1)
        if inference_mode:
            assert additional_input is not None
            inference_input = torch.cat([input, additional_input], dim=1)
            inference_gaussian_paramter = self.inference_linear(inference_input)
            inference_gaussian_paramter = torch.clamp(inference_gaussian_paramter, -4, 4)
            inference_mu, inference_logvar = torch.chunk(inference_gaussian_paramter, 2, 1)
            z = VariationalAutoEncoder.reparameter(inference_mu, inference_logvar, random_variable)
            kld = gaussian_kld(inference_mu, inference_logvar, prior_mu, prior_logvar)
            return z, kld
        else:
            z = VariationalAutoEncoder.reparameter(
                prior_mu, prior_logvar, random_variable)
            return z
