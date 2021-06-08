import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from metrics import calculate_accuracy, lehvenstein_distance

from tqdm import tqdm
import datetime as dt

WANDB_API_KEY= '52c84ab3f3b5c1f999c7f5f389f5e423f46fc04a'
import wandb
wandb.login()


def to_one_hot(labels, d):
    one_hot = torch.FloatTensor(labels.shape[0], d).cuda()
    one_hot.zero_()
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return one_hot


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):

        super().__init__(in_features, out_features, bias)

        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    def __init__(self, input_shape, d, hidden_size=[512, 512, 512],
                 ordering=None, one_hot_input=False):

        super().__init__()

        self.input_shape = input_shape
        self.nin = np.prod(input_shape)
        self.nout = self.nin * d
        self.d = d
        self.hidden_sizes = hidden_size
        self.ordering = np.arange(self.nin) if ordering is None else ordering
        self.one_hot_input = one_hot_input

        # define a simple MLP neural net
        self.net = []
        hs = [self.nin * d if one_hot_input else self.nin] + \
            self.hidden_sizes + [self.nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        self.m = {}
        self.create_mask()  # builds the initial self.m connectivity

    def create_mask(self):
        L = len(self.hidden_sizes)

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = self.ordering
        for l in range(L):
            self.m[l] = np.random.randint(self.m[l - 1].min(),
                                          self.nin - 1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :]
                 for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        masks[-1] = np.repeat(masks[-1], self.d, axis=1)
        if self.one_hot_input:
            masks[0] = np.repeat(masks[0], self.d, axis=0)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.one_hot_input:
            x = x.long().contiguous().view(-1)
            x = to_one_hot(x, self.d)
            x = x.view(batch_size, -1)
        else:
            x = x.float()
            x = x.view(batch_size, self.nin)
        logits = self.net(x).view(batch_size, self.nin, self.d)
        return logits.permute(0, 2, 1).contiguous().view(batch_size, self.d, *self.input_shape)

    def loss(self, x):
        return F.cross_entropy(self(x), x.long())

    def sample(self, n):
        samples = torch.zeros(n, self.nin).cuda()
        self.inv_ordering = {x: i for i, x in enumerate(self.ordering)}
        with torch.no_grad():
            for i in range(self.nin):
                logits = self(samples).view(n, self.d, self.nin)[
                    :, :, self.inv_ordering[i]]
                probs = F.softmax(logits, dim=1)
                samples[:, self.inv_ordering[i]] = torch.multinomial(
                    probs, 1).squeeze(-1)
            samples = samples.view(n, *self.input_shape)
        return samples.cpu().numpy().astype(int)

    def get_distribution(self):
        assert self.input_shape == (2,), 'Only available for 2D joint'
        x = np.mgrid[0:self.d, 0:self.d].reshape(2, self.d ** 2).T
        x = torch.LongTensor(x).cuda()
        log_probs = F.log_softmax(self(x), dim=1)
        distribution = torch.gather(log_probs, 1, x.unsqueeze(1)).squeeze(1)
        distribution = distribution.sum(dim=1)
        return distribution.exp().view(self.d, self.d).detach().cpu().numpy()


def train(model, loader, optimizer, device):
    """One epoch of training."""

    model.train()

    losses = []
    for source in loader:

                # source, target shapes: (B, T+1)
        source = source.to(device)

        optimizer.zero_grad()

        loss = model.loss(source)
        loss.backward()

        optimizer.step()
        loss.detach()
        losses.append(loss.item())

    return np.mean(losses)


def test(model, loader, device):

    model.eval()

    losses = []
    for source in loader:
        with torch.no_grad():

            # source, target shapes: (B, T+1)
            source = source.to(device)

            loss = model.loss(source)
            losses.append(loss.item())

    return np.mean(losses)


def main(model, optimizer, device, train_loader, validation_loader, N_epochs, return_best=False):

    model_name = dt.datetime.strftime(dt.datetime.now(),"%d_%m__%H_%M")
    from copy import deepcopy
    # print(samples.shape)
    min_val_loss = float('inf')

    for epoch in tqdm(range(N_epochs)):

        train_loss = train(model, train_loader, optimizer, device)
        validation_loss = test(model, validation_loader, device)

        if not (epoch % 25):
            print('Epoch: {}, train_loss: {:.6f}, validation_loss: {:.6f}'.format(epoch+1, train_loss, validation_loss))
            samples = model.sample(10)
            print(samples[0])

        if return_best and validation_loss < min_val_loss:
            best_model = deepcopy(model)
            min_val_loss = validation_loss
            checkpoint(model_name, model, optimizer, epoch)

    if return_best:
        return best_model

def main_wandb(model, optimizer, device, train_loader, validation_loader, params, project, return_best=False):

    N_epochs = params['training']['N_epochs']

    model_name = dt.datetime.strftime(dt.datetime.now(),"%d_%m__%H_%M")
    from copy import deepcopy

    min_val_loss = float('inf')

    with wandb.init(project=project, name=model_name, config=params, entity='nbg'):
        
        wandb.watch(model, log='all')

        for epoch in tqdm(range(N_epochs)):

            train_loss = train(model, train_loader, optimizer, device)
            validation_loss = test(model, validation_loader, device)

            wandb.log({'train_loss': train_loss,
                    'validation_loss': validation_loss})

            if not (epoch % 25):
                print('Epoch: {}, train_loss: {:.6f}, validation_loss: {:.6f}'.format(epoch+1, train_loss, validation_loss))
                samples = model.sample(10)
                print(samples[0])
                wandb.log({'samples': samples})

            if return_best and validation_loss < min_val_loss:
                best_model = deepcopy(model)
                min_val_loss = validation_loss
                checkpoint(model_name, model, optimizer, epoch)

    if return_best:
        return best_model


def checkpoint(model_name, model, optimizer, epoch):

    model_path = os.path.join('model_checkpoints', model_name+'.pt')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, model_path)
