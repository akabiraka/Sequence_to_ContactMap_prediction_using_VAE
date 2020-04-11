import torch
import torch.nn as nn
from torch.autograd import Variable
import constants as CONSTANTS

# num of input channels
nc = 1
# size of latent space
nz = 50
# Size of feature maps in encoder
nef = 64
# Size of feature maps in decoder
ndf = 64
# window size
w = CONSTANTS.WINDOW_SIZE


class BasicVAE1(nn.Module):
    """docstring for BasicVAE1."""

    def __init__(self):
        super(BasicVAE1, self).__init__()
        self.have_cuda = True if torch.cuda.is_available() else False

        self.fc1 = nn.Linear(2 * w, 20)

        self.encoder = nn.Sequential(
            # input: nc x 20 x 20; 28->CONSTANTS.WINDOW_SIZE
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: nef x 10 x 10
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*2) x 5 x 5
            nn.Conv2d(nef * 2, nef * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*4) x 3 x 3
            nn.Conv2d(nef * 4, 1024, 3, 2, 0, bias=False),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
            # nn.Sigmoid()
            # state size: 1024 x 1 x 1
        )

        self.fc2 = nn.Linear(1024, 512)
        self.fc31 = nn.Linear(512, nz)  # mean, mu
        self.fc32 = nn.Linear(512, nz)  # variance, var

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(1024, ndf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            # state size: (nef*8) x 4 x 4
            nn.ConvTranspose2d(ndf * 8, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(True),
            # state size. (ndf*4) x 8 x 8
            nn.ConvTranspose2d(ndf * 4, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(True),
            # # state size. (ndf*2) x 16 x 16
            nn.ConvTranspose2d(ndf * 2, nc, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ndf),
            # nn.ReLU(True),
            # state size. (nc) x 32 x 32
            # nn.ConvTranspose2d(ndf, nc, 4, 2, 1, bias=False),
            # nn.Tanh()
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

        self.fc4 = nn.Linear(nz, 512)
        self.fc5 = nn.Linear(512, 1024)

        self.relu = nn.ReLU()

    def encode(self, x):
        conv = self.encoder(x)
        # print("encode:", conv.size())
        h1 = self.fc2(conv.view(-1, 1024))
        # print("encode h1:", h1.size())
        mu = self.fc31(h1)  # mean
        logvar = self.fc32(h1)  # variance
        # print("mean:", mu.size(), "variance:", var.size())
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.have_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.relu(self.fc4(z))
        deconv_input = self.fc5(h3)
        # print("deconv_input:", deconv_input.size())
        deconv_input = deconv_input.view(-1, 1024, 1, 1)
        # print("deconv_input:", deconv_input.size())
        return self.decoder(deconv_input)

    def forward(self, x):
        x = self.fc1(x)
        print("occurance matric:", x.size())
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        print("z:", z.size())
        decoded = self.decode(z)
        print("decoded:", decoded.size())
        return decoded, mu, logvar
