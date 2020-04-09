import torch
import torch.nn as nn
from torch.autograd import Variable

# num of input channels
nc = 1
# size of latent space
nz = 20
# Size of feature maps in encoder
nef = 64
# Size of feature maps in decoder
ndf = 64


class BasicVAE(nn.Module):
    """docstring for BasicVAE."""

    def __init__(self):
        super(BasicVAE, self).__init__()
        self.have_cuda = True if torch.cuda.is_available() else False
        self.encoder = nn.Sequential(
            # input: nc x 28 x 20; 28->CONSTANTS.WINDOW_SIZE
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: nef x 14 x 10
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*2) x 7 x 5
            nn.Conv2d(nef * 2, nef * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size: (nef*4) x 4 x 3
            nn.Conv2d(nef * 4, 1024, 3, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
            # state size: 1024 x 2 x 1
            # nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(1024, ndf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(True),
            # state size. (ndf*8) x 4 x 4
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
            # nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

        self.fc1 = nn.Linear(1024 * 2 * 1, 512)
        self.fc21 = nn.Linear(512, nz)
        self.fc22 = nn.Linear(512, nz)

        self.fc3 = nn.Linear(nz, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        conv = self.encoder(x)
        # print("encode:", conv.size())
        h1 = self.fc1(conv.view(-1))
        # print("encode h1:", h1.size())
        mu = self.fc21(h1)  # mean
        var = self.fc22(h1)  # variance
        # print("mean:", mu.size(), "variance:", var.size())
        return mu, var

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        # print("deconv_input:", deconv_input.size())
        deconv_input = deconv_input.view(-1, 1024, 1, 1)
        # print("deconv_input:", deconv_input.size())
        return self.decoder(deconv_input)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.have_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, x1, x2):
        print("x1: ", x1.size())
        mu1, var1 = self.encode(x1)
        z1 = self.reparametrize(mu1, var1)
        print("z1:", z1.size())
        mu2, var2 = self.encode(x1)
        z2 = self.reparametrize(mu1, var1)
        print("z2:", z2.size())
        z = (z1 + z2) / 2
        print("z:", z.size())
        decoded = self.decode(z)
        print("decoded:", decoded.size())
        return decoded
