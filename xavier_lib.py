
import torch
import numpy as np


def compute_statistic_and_update(samples, sum_mean, sum_covar, counter) -> None:
    samples_num = list(samples.shape)[0]
    counter += samples_num
    if sum_mean is not None:
        sum_mean += torch.sum(samples, dim=0)
    sum_covar += torch.mm(samples.permute(1, 0), samples)


class ForwardStatisticHook(object):

    def __init__(self, name=None, dim=4):
        self.name = name
        self.dim = dim
        self.sum_mean = None
        self.sum_covariance = None
        self.counter = None

    def __call__(self, module, inputs, output) -> None:
        with torch.no_grad():
            channel_num = list(inputs[0].shape)[1]
            if self.sum_mean is None or self.sum_covariance is None:
                self.sum_mean = torch.nn.Parameter(torch.zeros(channel_num)).cuda()
                self.sum_covariance = torch.nn.Parameter(torch.zeros(channel_num, channel_num)).cuda()
                self.counter = torch.nn.Parameter(torch.zeros(1)).cuda()
                print('forward:', self.name, self.sum_covariance.shape)
            # from [N,C,W,H] to [N*W*H,C]
            if self.dim == 4:
                samples = inputs[0].permute(0, 2, 3, 1).contiguous().view(-1, channel_num)
            else:
                samples = inputs[0]
            compute_statistic_and_update(samples, self.sum_mean, self.sum_covariance, self.counter)


class BackwardStatisticHook(object):

    def __init__(self, name=None, dim=4):
        self.name = name
        self.dim = dim
        self.sum_covariance = None
        self.counter = None

    def __call__(self, module, grad_input, grad_output) -> None:
        channel_num = list(grad_output[0].shape)[1]
        if self.sum_covariance is None:
            self.sum_covariance = torch.nn.Parameter(torch.zeros(channel_num, channel_num)).cuda()
            self.counter = torch.nn.Parameter(torch.zeros(1)).cuda()
            print('backward:', self.name, self.sum_covariance.shape)
        if self.dim == 4:
            samples = grad_output[0].permute(0, 2, 3, 1).contiguous().view(-1, channel_num)
        else:
            samples = grad_output[0]
        compute_statistic_and_update(samples, None, self.sum_covariance, self.counter)


class StatisticManager(object):

    def __init__(self):

        self.name_to_statistic = {}

    def __call__(self, model):

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, torch.nn.Conv2d):
                if sub_module.kernel_size[0] == 1:
                    hook_cls = ForwardStatisticHook(name)
                    back_hook_cls = BackwardStatisticHook(name)
                    sub_module.register_forward_hook(hook_cls)
                    sub_module.register_backward_hook(back_hook_cls)
                    self.name_to_statistic[name] = (hook_cls, back_hook_cls)
            elif isinstance(sub_module, torch.nn.Linear):
                hook_cls = ForwardStatisticHook(name, dim=2)
                back_hook_cls = BackwardStatisticHook(name, dim=2)
                sub_module.register_forward_hook(hook_cls)
                sub_module.register_backward_hook(back_hook_cls)
                self.name_to_statistic[name] = (hook_cls, back_hook_cls)

    def visualize(self):
        