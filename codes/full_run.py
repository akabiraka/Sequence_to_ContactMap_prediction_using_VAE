import os
import math
import numpy as np
import matplotlib.pyplot as plt
# from tqdm.notebook import tqdm
from progress.bar import Bar

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from protein_dataset import ProteinDataset
import constants as CONSTANTS
from models.basic_vae_1 import BasicVAE1
from models.vae_loss import VAELoss


# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BasicVAE1()
model.to(device)
criterion = VAELoss()
init_lr = 1e-7
optimizer = optim.Adam(model.parameters(), lr=init_lr)
batch_size = 40
n_epochs = 100
print_every = 1
test_every = 4
plot_every = 5
print("device=", device)
print("batch_size=", batch_size)
print("n_epochs=", n_epochs)
print("init_lr=", init_lr) 
print("loss=BCE-sum")
print(model)

print("loading training dataset ... ...")
train_dataset = ProteinDataset(CONSTANTS.TRAIN_FILE)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print("train dataset len: ", train_dataset.__len__())
# x, y = train_dataset.__getitem__(0)
# print(x.shape, y.shape)
print("train loader size: ", len(train_loader))
print("successfully loaded training dataset ... ...")

print("loading validation dataset ... ...")
val_dataset = ProteinDataset(CONSTANTS.VAL_FILE)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
print("val dataset len:", val_dataset.__len__())
print("val loader size: ", len(val_loader))
print("successfully loaded validation dataset ... ...")

# test_dataset = ProteinDataset(CONSTANTS.TEST_FILE)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
# print(len(test_loader))


def train():
    model.train()
    loss = 0.0
    losses = []
    n_train = len(train_loader)
    bar = Bar('Processing training:', max=n_train)
    for i, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # print("x:", x.shape, "y:", y.shape)
        optimizer.zero_grad()
        y_prime, mu, logvar = model(x)
        # y_prime.squeeze_(0)
        # print("y_prime:", y_prime.size(), "y:", y.size())
        loss = criterion(y, y_prime, mu, logvar)
        # print(loss)
        loss.backward()
        optimizer.step()
        losses.append(loss)
        # if i != 0 and i % 20 == 0:
        #     print("training {}/{} th batch | loss: {:.5f}".format(i, n_train, loss.item()))
        bar.next()
    bar.finish()
    return torch.stack(losses).mean().item()

# train()


def test(data_loader):
    model.eval()
    loss = 0.0
    losses = []
    n_test = len(data_loader)
    test_bar = Bar('Processing testing:', max=n_test)
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            y_prime, mu, logvar = model(x)
            loss = criterion(y, y_prime, mu, logvar)
            losses.append(loss)
            test_bar.next()
        test_bar.finish()
        # if i != 0 and i % 20 == 0:
        #     print("testing {}/{} th batch | loss: {:.5f}".format(i, n_test, loss.item()))
    return torch.stack(losses).mean().item()

# test(test_loader)
# test(val_loader)


train_losses = []
val_losses = []
best_test_loss = np.inf
epoch_bar = Bar('Processing epochs:', max=(n_epochs + 1))
for epoch in range(1, n_epochs + 1):
    print("Starting epoch {}/{}".format(epoch, n_epochs + 1))

    train_loss = train()
    train_losses.append(train_loss)

    if epoch % print_every == 0:
        print("epoch:{}/{}, train_loss: {:.7f}".format(epoch, n_epochs + 1, train_loss))
        for param_group in optimizer.param_groups:
            print("learning rate: ", param_group['lr'])

    if epoch % test_every == 0:
        print("Starting testing epoch {}/{}".format(epoch, n_epochs + 1))
        val_loss = test(val_loader)
        print("epoch:{}/{}, val_loss: {:.7f}".format(epoch, n_epochs + 1, val_loss))
        val_losses.append(val_loss)
        if val_loss < best_test_loss:
            best_test_loss = val_loss
            print('Updating best val loss: {:.7f}'.format(best_test_loss))
            torch.save(model.state_dict(), '../outputs/best_model_26.pth')

    if epoch % plot_every == 0:
        pass
        # plt.plot(train_losses)
        # plt.plot(val_losses)
        # plt.show()
        # plt.savefig("outputs/raw_img_with_raw_pt_2_{}.jpg".format(epoch))

    epoch_bar.next()
epoch_bar.finish()

print(train_losses)
print(val_losses)
