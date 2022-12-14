# PyTorch Second-order Optimizers

PyTorch implementations of easy-to-use and memory-efficient second-order optimizers.

The following is a list of currently supported optimizers (ongoing):

- Hessian-free Levenberg&ndash;Marquardt (HFLM) \
  Usage: `python3 test_cifar_10_classification.py --optim lm`
- Hessian-free Gauss&ndash;Newton (HFGN) \
  Usage: `python3 test_cifar_10_classification.py --optim gn`
- Stochastic gradient descent (SGD) with Nesterov momentum \
  Usage: `python3 test_cifar_10_classification.py --optim sgd`

You can use `python3 test_cifar_10_classification.py --help` to see the list of available arguments.

Mathematical details and documentation coming soon.

Currently test on CNNs has been implemented, and test on GANs are coming soon.