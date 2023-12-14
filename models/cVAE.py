import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torchvision.utils import make_grid


class CondVAE(nn.Module):
    def __init__(self,
                 recognition,
                 generation,
                 conditional_prior,
                 condition_feature_extractor,
                 weight_kl_div=1.0,
                 learn_sigma=False,
                 **args):
        
        super(CondVAE, self).__init__()
        
        self.recognition = recognition
        self.generation = generation
        self.conditional_prior = conditional_prior
        self.condition_feature_extractor = condition_feature_extractor
        self.weight_kl_div = weight_kl_div
        sigma = 1.0
        if learn_sigma:
            sigma = nn.Parameter(torch.tensor(sigma, dtype=torch.float))
            self.register_parameter("sigma", sigma)
        else:
            self.sigma = torch.tensor(sigma, dtype=torch.float)

    def sample_latent(self, z):
        half_chan = int(z.shape[1] / 2)
        mu, log_sig = z[:, :half_chan], z[:, half_chan:]
        eps = torch.randn(*mu.shape, dtype=torch.float32)
        eps = eps.to(z.device)
        return mu + torch.exp(log_sig) * eps

    def train_step(self, x, y, optimizer, **kwargs):

        optimizer.zero_grad()

        z1 = self.recognition(torch.cat([x, y], dim=1))
        z2 = self.conditional_prior(x)
        condition_feature = self.condition_feature_extractor(x)
        z1_sample = self.sample_latent(z1)
        pred = self.generation(torch.cat([z1_sample, condition_feature], dim=1))
        
        nll = self.nll(pred, y)
        kl_loss = self.kl_loss(z1, z2)

        loss = nll + kl_loss * self.weight_kl_div

        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
            "train_nll": nll.item(),
            "train_kl_loss": kl_loss.item(),
            "partial_map@": make_grid(x),
            "full_map@": make_grid(y),
            "predicted_map@": make_grid(torch.round(pred))
            }
        
    def validation_step(self, x, y, **kwargs):
        with torch.no_grad():
            z1 = self.recognition(torch.cat([x, y], dim=1))
            z2 = self.conditional_prior(x)
            condition_feature = self.condition_feature_extractor(x)
            z1_sample = self.sample_latent(z1)
            pred = self.generation(torch.cat([z1_sample, condition_feature], dim=1))
            
            nll = self.nll(pred, y)
            kl_loss = self.kl_loss(z1, z2)

            loss = nll + kl_loss * self.weight_kl_div
            
        return {
            "loss": loss.item(),
            "valid_nll": nll.item(),
            "valid_kl_loss": kl_loss.item(),
            "partial_map@": make_grid(x),
            "full_map@": make_grid(y),
            "predicted_map@": make_grid(torch.round(pred)),
            }
    
    def sample(self, x, sample_num=10):
        with torch.no_grad():
            batch_num, channel_num, width, height = x.shape
            z = self.conditional_prior(x)
            condition_feature = self.condition_feature_extractor(x)
            z = z.repeat_interleave(sample_num, dim=0)
            condition_feature = condition_feature.repeat_interleave(sample_num, dim=0)
            z_sample = self.sample_latent(z)
            pred = self.generation(torch.cat([z_sample, condition_feature], dim=1))
            pred = pred.reshape(batch_num, sample_num, channel_num, width, height)
        return {"partial_map@": make_grid(x),
                "grid@": make_grid(pred.reshape(batch_num * sample_num, channel_num, width, height), nrow=sample_num)
                }
            
    def kl_loss(self, z1, z2):
        """analytic (positive) KL divergence between gaussians
        KL(q(z|x, y) | p(z|x))"""
        half_chan = int(z1.shape[1] / 2)
        mu1, log_sig1 = z1[:, :half_chan], z1[:, half_chan:]
        mu2, log_sig2 = z2[:, :half_chan], z2[:, half_chan:]
        sig_sq1, sig_sq2 = torch.exp(log_sig1) ** 2 + 1e-5, torch.exp(log_sig2) ** 2 + 1e-5
        kl = torch.log(sig_sq2) - torch.log(sig_sq1) - 1 + (mu1 - mu2) ** 2 / sig_sq2 + sig_sq1 / sig_sq2
        return 0.5 * torch.sum(kl, dim=1).mean() # kl.view(len(kl), -1)
    
    def nll(self, pred, y):
        D = torch.prod(torch.tensor(y.shape[1:]))
        sig = self.sigma
        const = -D * 0.5 * torch.log(
            2 * torch.tensor(np.pi, dtype=torch.float32)
        ) - D * sig
        loglik = const - 0.5 * ((y - pred) ** 2).view((y.shape[0], -1)).sum(
            dim=1
        ) / (torch.exp(sig) ** 2)
        return -loglik.mean()
    
