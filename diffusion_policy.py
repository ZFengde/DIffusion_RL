# Import of libraries
import random
from argparse import ArgumentParser

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST, FashionMNIST
from torchvision.transforms import Compose, Lambda, ToTensor
from tqdm.auto import tqdm # this is for generating training bar

'''
Now,  𝜖𝜃  is a function of both  𝑥  and  𝑡  
and we don't want to have a distinct model for 
each denoising step (thousands of independent models)
but instead we want to use a single model that 
takes as input the image  𝑥  
and the scalar value indicating the timestep  𝑡.

To do so, in practice we use a sinusoidal embedding (function sinusoidal_embedding) 
that maps each time-step to a time_emb_dim dimension. 
These time embeddings are further mapped with some time-embedding MLPs (function _make_te) 
and added to tensors through the network in a channel-wise manner.
'''
def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    # n, d = n_steps, time_emb_dim
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)]) # a^x
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, ::2])

    return embedding


# DDPM class
class DDPM(nn.Module):
    def __init__( # these parameters are predefined and fixed
        self,
        n_steps=200,
        # beta that tells how much we are distorting the image in that step
        min_beta=10**-4, # value of the beta_1
        max_beta=0.02, # value of the beta_T
        device=None,
        image_chw=(1, 28, 28), # tuple contining dimensionality of images
    ):
        super(DDPM, self).__init__()
        self.n_steps = n_steps # number of diffusion steps
        self.device = device
        self.image_chw = image_chw
        self.network = nn.Sequential().to(device)

        '''
        So here for generating coefficients w.r.t. noise
        '''
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device # so this is a array of scalar start from min_beta to max_beta with n_steps
        )  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas # alpha_t = 1 - beta_t | temporal coefficient

        # coefficient starting from zero, use cumprod basically
        # could be replaced to: 
        # self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)
        self.alpha_bars = torch.tensor(
            [torch.prod(self.alphas[: i + 1]) for i in range(len(self.alphas))]
        ).to(device)

    def forward(self, x0, t, eta=None): # get noisy images
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape

        # pick out the coefficient regarding the specific time step
        a_bar = self.alpha_bars[t] 

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device) # normal distribution, indepdent for every pixel

        noisy = (
            a_bar.sqrt().reshape(n, 1, 1, 1) * x0 # should be expand?
            + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        )
        return noisy # this the image with added noise

    def backward(self, x, t): # predict noise
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.

        return self.network(x, t) # this is just the noise

class Networks(nn.Module):
    def __init__(self, n_steps=100, time_emb_dim=100, env_space=100): # n_steps=env_max_episode_steps
        super(Networks, self).__init__()
        # Sinusoidal embedding, keep it as a fixed look-up table, 
        # by inputting indexes, we can get the corresponding time-wise embedding we want
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        self.time_embeddings_1 = self._make_te(time_emb_dim, env_space) # making time embedding into the same dimension as env_space
        self.networks_1 = nn.Sequential(
            nn.Linear(env_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
        )

        self.time_embeddings_2 = self._make_te(time_emb_dim, 64) 
        self.networks_2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, env_space),
        )

    def forward(self, x, t):
        # the input t here is an array of timestep
        t = self.time_embed(t) # the output t is an initial time embedding, should be timestep indicator
        # actually in every block we take that as input to generate a block-wise time embedding

        x = self.networks_1(x + self.time_embeddings_1(t)) 
        output = self.networks_2(x + self.time_embeddings_2(t)) 

        return output 
        # prediction of the noise added on the x based on the given timestep t

    # make time embedding
    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out)
        )

def generate_new_samples( # this is for generating new samples based on trained model
    ddpm,
    n_samples=16,
    device=None,
    obs_space = 10,
    action_space = 10,
    reward_space = 1,
    size = 0,
):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, obs_space + action_space + reward_space).to(device) # this is random gaussian noise

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # get t from the final timestep to the start
            # Estimating noise to be removed
            # get several sample
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor) # estimate of the noise, all the way back

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            # corresponding to the mu_theta in original blog
            # so this is a sampled noisy image based on DDPM model and timestep
            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
            ) # /mu(x_{t-1}) prediction through model

            if t > 0: # this is random noise term
                z = torch.randn(n_samples, size).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

            # Adding frames to the GIF

    return x

def training_loop(
    ddpm, loader, n_epochs, optim, device, display=False, store_path="ddpm_model.pt"
):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(
            tqdm(
                loader,
                leave=False,
                desc=f"Epoch {epoch + 1}/{n_epochs}",
                colour="#005500",
            )
        ):
            # Loading data
            x0 = batch[0].to(device)
            n = len(x0) # batch size

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(device) # this could be eta = a = pi(s)
            t = torch.randint(0, n_steps, (n,)).to(device) # generate n different number of random time t

            # Computing the noisy image based on x0 and the time-step (forward process)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step
            eta_theta = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_theta, eta)
            
            optim.zero_grad()
            loss.backward()
            optim.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += " --> Best model ever (stored)"

        print(log_string)
