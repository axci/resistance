import torch
import numpy as np
import math


def check_array(input_array) -> torch.Tensor:
    """
    Convert array to a tensor.
    
    Args:
        input_array: list / np.array / torch.Tensor

    Returns:
        torch.Tensor: The input converted to a PyTorch tensor.
    """
    if isinstance(input_array, torch.Tensor):
        return input_array
    elif isinstance(input_array, np.ndarray):
        return torch.from_numpy(input_array)
    else:
        input_array = np.array(input_array)
        return torch.from_numpy(input_array)
    
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:
        raise NotImplementedError("The action space should be either Discrete or MultiBinary")
    return act_shape

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'MultiBinary':
        obs_shape = obs_space.shape[0]
    elif obs_space.__class__.__name__ == "MultiDiscrete":
        obs_shape = obs_space.shape[0]
    else:
        raise NotImplementedError("The observation space should be either MultiDiscrete or MultiBinary")
    return obs_shape