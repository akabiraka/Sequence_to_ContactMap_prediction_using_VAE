import torch
import torch.nn as nn
import torch.nn.functional as F

import constants as CONSTANTS


class VAELoss(nn.Module):
    """docstring for VAELoss."""

    def __init__(self):
        super(VAELoss, self).__init__()
        self.w_size = CONSTANTS.WINDOW_SIZE

    def forward(self, y, y_prime, mu, logvar):
        """
            reconstruction + KL divergence loss
        """
        print(y_prime.size(), y.size())
        y_prime = y_prime.view(-1, self.w_size * self.w_size)
        y = y.view(-1, self.w_size * self.w_size)

        BCE = F.binary_cross_entropy(y_prime, y, size_average=False)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # return BCE + KLD
        return BCE + 3 * KLD
