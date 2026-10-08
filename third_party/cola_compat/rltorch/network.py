"""create_linear_network as in ku2482/rltorch."""
import torch.nn as nn


def weights_init_xavier(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight, gain=1)
        nn.init.constant_(m.bias, 0)


def weights_init_he(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
        nn.init.constant_(m.bias, 0)


_INIT = {"xavier": weights_init_xavier, "he": weights_init_he}


def create_linear_network(input_dim, output_dim, hidden_units=[], hidden_activation="relu",
                          output_activation=None, initializer="xavier"):
    act = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid}
    model, units = [], input_dim
    for next_units in hidden_units:
        model.append(nn.Linear(units, next_units))
        model.append(act[hidden_activation]())
        units = next_units
    model.append(nn.Linear(units, output_dim))
    if output_activation is not None:
        model.append(act[output_activation]())
    return nn.Sequential(*model).apply(_INIT[initializer])
