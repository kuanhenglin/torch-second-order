import math
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from torchvision import datasets as dsets, transforms as trans
from tqdm import tqdm
from optimizers.gradient_descent import GradientDescent

from optimizers.gradient_descent import GradientDescent
from optimizers.levenberg_marquardt import LevenbergMarquardt
import optimizers.utilities as utils


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_arch():
    arch = dict(
        cnn_3=dict(filter=[16, 16, 16], kernel=[5, 3, 3],
                   max_pool=[2, 2, 2], batch_norm=False,
                   act="leaky_relu", act_out="linear"),
        cnn_4=dict(filter=[32, 64, 64], kernel=[5, 3, 3],
                   max_pool=[2, 2, 2], batch_norm=False,
                   act="relu", act_out="linear"),
        cnn_7=dict(filter=[32, 32, 64, 64, 64, 128], kernel=[5, 3, 3, 3, 3, 3],
                   max_pool=[1, 2, 1, 2, 1, 2], batch_norm=False,
                   act="leaky_relu", act_out="linear"))
    return arch


class CNN(nn.Module):

    def __init__(self, arch, in_shape, out_shape):
        super(CNN, self).__init__()

        self.arch = arch
        self.in_shape, self.out_shape = in_shape, out_shape  # assume NCHW format
        self.net, self.size = None, None

        self._init_net()
        self.apply(self._init_weights)

    def forward(self, inputs):
        return self.net(inputs)

    def _init_net(self):
        layers = []

        for i in range(len(self.arch["filter"])):
            in_filter = self.in_shape[0] if i == 0 else self.arch["filter"][i - 1]
            layers.append(nn.Conv2d(in_channels=in_filter, out_channels=self.arch["filter"][i],
                                    kernel_size=self.arch["kernel"][i], padding="same"))
            self._add_act(layers=layers, act=self.arch["act"])
            if self.arch["max_pool"][i] > 1:
                layers.append(nn.MaxPool2d(kernel_size=self.arch["max_pool"][i], ceil_mode=True))
            if self.arch["batch_norm"]:
                layers.append(nn.BatchNorm2d(num_features=self.arch["filter"][i]))

        layers.append(nn.Flatten())
        flatten_size = np.prod(self._flatten_shape())
        layers.append(nn.Linear(in_features=flatten_size, out_features=self.out_shape[0]))
        self._add_act(layers=layers, act=self.arch["act_out"])

        self.net = nn.Sequential(*layers)

        self.size = np.sum([np.prod(param.shape) for param in self.parameters()])

    def _add_act(self, layers, act):
        if act == "relu":
            layers.append(nn.ReLU())
        elif act == "leaky_relu":
            layers.append(nn.LeakyReLU(negative_slope=0.3))
        elif act == "softmax":
            layers.append(nn.Softmax())
        elif act == "linear":
            pass
        return layers

    def _flatten_shape(self):
        def flatten_size(dim, max_pool):
            if len(max_pool) == 0:
                return dim
            return flatten_size(dim=int(np.ceil(dim / max_pool[0])), max_pool=max_pool[1:])

        return (self.arch["filter"][-1],
                flatten_size(dim=self.in_shape[1], max_pool=self.arch["max_pool"]),
                flatten_size(dim=self.in_shape[2], max_pool=self.arch["max_pool"]))

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=0.3, nonlinearity=self.arch["act"])
            nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5.0))
            nn.init.constant_(module.bias, val=0.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
            nn.init.constant_(module.bias, val=0.0)


def evaluate(fn, data, batch):
    _, (outputs, loss) = utils.minibatch(None, fn, data, batch=batch)
    loss = loss.cpu().item()
    acc = (torch.max(outputs, dim=1)[1] == data[1]).to(torch.float32).mean().cpu().item()
    return loss, acc


def main():
    # argument parser
    parser = ArgumentParser(description="Hessian-free Levenberg-Marquardt optimizer.")
    parser.add_argument("-o", "--optim", dest="optim", default="lm",
                        choices=("lm", "gn", "sgd"),
                        help="Type of optimizer used, affects hyperparameters.")
    parser.add_argument("-n", "--net", dest="net", default="cnn_4",
                        choices=("cnn_3", "cnn_4", "cnn_7"),
                        help="Network architecture tested, see Python file for details.")
    parser.add_argument("-s", "--seed", dest="seed", default=0, type=int,
                        help="Seed for reproducability.")
    args = parser.parse_args()

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on device: {device}\n")

    # seed
    set_seed(args.seed)

    # model
    arch = get_arch()
    model = CNN(arch=arch[args.net], in_shape=(3, 32, 32), out_shape=(10,)).to(device)
    print(f"# of model parameters: {model.size}\n")

    # train data, feed entire dataset (for samping) every iteration
    train_data = dsets.CIFAR10(root='./data', train=True, download=False)
    train_data.data = torch.moveaxis(
        torch.tensor(train_data.data / 255, dtype=torch.float32, device=device), 3, 1)
    train_data.targets = torch.tensor(train_data.targets, device=device)
    train_mean = train_data.data.mean(dim=0)[None, :]
    train_data.data.sub_(train_mean)  # mean-center data
    # test data
    test_data = dsets.CIFAR10(root='./data', train=False, download=False)
    test_data.data = torch.moveaxis(
        torch.tensor(test_data.data / 255, dtype=torch.float32, device=device), 3, 1)
    test_data.targets = torch.tensor(test_data.targets, device=device)
    test_data.data.sub_(train_mean)

    # objective and optimizer
    obj = lambda outputs, labels: (labels - outputs).square().sum(dim=1).mean()  # sum of squares
    if args.optim in ("lm", "gn"):
        damping = dict(lm=1.0, gn=0.0)[args.optim]
        optim = LevenbergMarquardt(model.parameters(), sample=0.05, minibatch=2500,
                                   damping=damping, damping_update=dict(method="nielsen"),
                                   weight_decay=0.002)
    elif args.optim == "sgd":
        optim = GradientDescent(model.parameters(), lr=0.003, momentum=0.9, nesterov=True,
                                weight_decay=0.002)

    # forward function
    def forward_fn(inputs):
        return model(inputs)
    # objective function
    def obj_fn(outputs, labels):
        with torch.no_grad():  # one-hot encode labels
            labels = nn.functional.one_hot(
                labels.to(torch.int64), num_classes=outputs.size(1)).to(torch.float32)
        return obj(outputs, labels)  # we want squared error (no mean)

    # train and evaluation loop
    iter_total = dict(lm=128, gn=256, sgd=64000)[args.optim]
    for i in tqdm(range(iter_total), position=0):
        # train
        model.zero_grad()
        if args.optim in ("lm", "gn"):
            train_inputs, train_labels = train_data.data, train_data.targets
        elif args.optim == "sgd":
            train_inputs, train_labels = utils.sample_data(
                data=(train_data.data, train_data.targets), sample=128)
        optim.step(fn=(forward_fn, obj_fn), data=(train_inputs, train_labels))
        # evaluate
        if (i + 1) % round(iter_total / 128) == 0:
            with torch.no_grad():  # stop gradient accumulation to save memory
                # train
                train_loss, train_acc = evaluate(
                    (forward_fn, obj_fn), (train_inputs, train_labels), batch=5000)
                # test
                test_inputs, test_labels = test_data.data, test_data.targets
                test_loss, test_acc = evaluate(
                    (forward_fn, obj_fn), (test_inputs, test_labels), batch=5000)
                # write
                tqdm.write(f"[{i + 1}]\t   "
                           f"Train: {round(train_acc * 100, 3):.3f}% ({round(train_loss, 5):.5f})\t"
                           f"Test: {round(test_acc * 100, 3):.3f}% ({round(test_loss, 5):.5f})")

    print()


if __name__ == "__main__":
    main()