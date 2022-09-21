import numpy as np
import torch
from torch import nn
from torchvision import datasets as dsets, transforms as trans
from tqdm import tqdm
from optimizers.gradient_descent import GradientDescent

from optimizers.gradient_descent import GradientDescent
from optimizers.levenberg_marquardt import LevenbergMarquardt
import optimizers.utilities as utils


def get_arch():
    arch = dict(
        cnn_3=dict(filter=[16, 16, 16], kernel=[5, 3, 3],
                   max_pool=[2, 2, 2], batch_norm=False,
                   act="leaky_relu", act_out=None),
        cnn_4=dict(filter=[32, 64, 64], kernel=[5, 3, 3],
                   max_pool=[2, 2, 2], batch_norm=False,
                   act="leaky_relu", act_out=None),
        cnn_7=dict(filter=[32, 32, 64, 64, 64, 128], kernel=[5, 3, 3, 3, 3, 3],
                   max_pool=[1, 2, 1, 2, 1, 2], batch_norm=False,
                   act="leak_relu", act_out=None))
    return arch


class CNN(nn.Module):

    def __init__(self, arch, in_shape, out_shape):
        super(CNN, self).__init__()

        self.arch = arch
        self.in_shape, self.out_shape = in_shape, out_shape  # assume NCHW format
        self.net, self.size = None, None

        self._init_net()

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
        return layers

    def _flatten_shape(self):
        def flatten_size(dim, max_pool):
            if len(max_pool) == 0:
                return dim
            return flatten_size(dim=int(np.ceil(dim / max_pool[0])), max_pool=max_pool[1:])

        return (self.arch["filter"][-1],
                flatten_size(dim=self.in_shape[1], max_pool=self.arch["max_pool"]),
                flatten_size(dim=self.in_shape[2], max_pool=self.arch["max_pool"]))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on device: {device}\n")

    # model
    arch = get_arch()
    model = CNN(arch=arch["cnn_4"], in_shape=(3, 32, 32), out_shape=(10,)).to(device)
    print(f"# of model parameters: {model.size}\n")

    # train data, feed entire dataset (for samping) every iteration
    train_data = dsets.CIFAR10(root='./data', train=True, download=False)
    train_data.data = torch.moveaxis(
        torch.tensor(train_data.data / 255, dtype=torch.float32, device=device), 3, 1)
    train_data.targets = torch.tensor(train_data.targets, device=device)
    train_mean = train_data.data.mean(dim=0)[None, :]
    train_data.data.sub_(train_mean)  # mean-center data
    # test data
    test_set = dsets.CIFAR10(root='./data', train=False, download=False,
                             transform=trans.Compose([trans.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=5000)

    # objective and optimizer
    obj = lambda outputs, labels: (labels - outputs).square().sum(dim=1).mean()
    optim = LevenbergMarquardt(model.parameters(), sample=0.05, minibatch=2500, weight_decay=0.002)
    # optim = GradientDescent(model.parameters(), lr=0.003, momentum=0.9, nesterov=True,
    #                         weight_decay=0.002)

    # forward function
    def forward_fn(inputs):
        return model(inputs)
    # objective function
    def obj_fn(outputs, labels):
        with torch.no_grad():  # one-hot encode labels
            labels = nn.functional.one_hot(
                labels.to(torch.int64), num_classes=outputs.size(1)).to(torch.float32)
        return obj(outputs, labels)  # we want squared error (no mean)

    for i in tqdm(range(128), position=0):
        # train
        model.zero_grad()
        train_inputs, train_labels = train_data.data, train_data.targets
        # train_inputs, train_labels = utils.sample_data(
        #     data=(train_data.data, train_data.targets), sample=100)
        train_outputs, train_loss = optim.step(fn=(forward_fn, obj_fn),
                                               data=(train_inputs, train_labels))
        train_loss = train_loss.cpu().item()
        train_acc = (torch.max(train_outputs, dim=1)[1] == train_labels)\
            .to(torch.float32).mean().cpu().item()
        # test
        test_acc, test_loss = 0, 0
        if (i + 1) % 1 == 0:
            with torch.no_grad():  # stop gradient accumulation to save memory
                for j, data in enumerate(test_loader):
                    test_inputs, test_labels = data[0].to(device), data[1].to(device)
                    test_inputs.sub_(train_mean)
                    test_outputs = model(test_inputs)
                    test_loss += obj_fn(test_outputs, test_labels) *\
                        test_inputs.size(0) / len(test_set)
                    test_acc += (torch.max(test_outputs, dim=1)[1] == test_labels)\
                        .to(torch.float32).mean() * test_inputs.size(0) / len(test_set)
                test_acc, test_loss = test_acc.cpu().item(), test_loss.cpu().item()
                tqdm.write(f"[{i + 1}]\t   "
                           f"Train: {(train_acc * 100):3f}% ({train_loss:5f})\t"
                           f"Test: {(test_acc * 100):3f}% ({test_loss:5f})")

    print()


if __name__ == "__main__":
    main()