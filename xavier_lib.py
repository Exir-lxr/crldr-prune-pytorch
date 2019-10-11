import torch
import numpy as np


class InfoStruct(object):

    def __init__(self, module, f_cls, b_cls):
        self.module = module
        self.f_cls = f_cls
        self.b_cls = b_cls


def compute_statistic_and_update(samples, sum_mean, sum_covar, counter) -> None:
    samples = samples.to(torch.half).to(torch.double)
    samples_num = list(samples.shape)[0]
    counter += samples_num
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
                self.sum_mean = torch.nn.Parameter(torch.zeros(channel_num).to(torch.double),
                                                   requires_grad=False).cuda()
                self.sum_covariance = \
                    torch.nn.Parameter(torch.zeros(channel_num, channel_num).to(torch.double),
                                       requires_grad=False).cuda()
                self.counter = torch.nn.Parameter(torch.zeros(1).to(torch.double), requires_grad=False).cuda()
            # from [N,C,W,H] to [N*W*H,C]
            if self.dim == 4:
                samples = inputs[0].permute(0, 2, 3, 1).contiguous().view(-1, channel_num)
            else:
                samples = inputs[0]
            compute_statistic_and_update(samples, self.sum_mean, self.sum_covariance, self.counter)

            #if self.name == 'blocks.0.0.conv_pw':
            #    print(samples[:,1])
            #    for_write = list(np.array(samples.cpu()))
            #    with open('check.txt', 'a') as f:
            #        for line in for_write:
            #            for w in line:
            #                f.write(str(w)+', ')
            #            f.write('\n')


class BackwardStatisticHook(object):

    def __init__(self, name=None, dim=4):
        self.name = name
        self.dim = dim
        self.sum_covariance = None
        self.sum_mean = None
        self.counter = None

    def __call__(self, module, grad_input, grad_output) -> None:
        with torch.no_grad():
            channel_num = list(grad_output[0].shape)[1]
            if self.sum_covariance is None:
                self.sum_mean = torch.nn.Parameter(torch.zeros(channel_num).to(torch.double),
                                                   requires_grad=False).cuda()
                self.sum_covariance = \
                    torch.nn.Parameter(torch.zeros(channel_num, channel_num).to(torch.double),
                                       requires_grad=False).cuda()
                self.counter = torch.nn.Parameter(torch.zeros(1).to(torch.double), requires_grad=False).cuda()
                # print('backward:', self.name, self.sum_covariance.shape)
            if self.dim == 4:
                samples = grad_output[0].permute(0, 2, 3, 1).contiguous().view(-1, channel_num)
            else:
                samples = grad_output[0]
            compute_statistic_and_update(samples, self.sum_mean, self.sum_covariance, self.counter)


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
                    self.name_to_statistic[name] = InfoStruct(sub_module, hook_cls, back_hook_cls)
            elif isinstance(sub_module, torch.nn.Linear):
                hook_cls = ForwardStatisticHook(name, dim=2)
                back_hook_cls = BackwardStatisticHook(name, dim=2)
                sub_module.register_forward_hook(hook_cls)
                sub_module.register_backward_hook(back_hook_cls)
                self.name_to_statistic[name] = InfoStruct(sub_module, hook_cls, back_hook_cls)

    def visualize(self):

        from matplotlib import pyplot as plt
        i = 1
        for name in self.name_to_statistic:
            info = self.name_to_statistic[name]
            forward_mean = info.f_cls.sum_mean / info.f_cls.counter
            forward_cov = (info.f_cls.sum_covariance / info.f_cls.counter) - \
                torch.mm(forward_mean.view(-1, 1), forward_mean.view(1, -1))

            grad_mean = info.b_cls.sum_mean / info.b_cls.counter
            grad_cov = (info.b_cls.sum_covariance / info.b_cls.counter) - \
                torch.mm(grad_mean.view(-1, 1), grad_mean.view(1, -1))
            plt.subplot(10, 15, i)
            plt.imshow(np.array(forward_cov.cpu()), cmap='hot')
            plt.xticks([])
            plt.yticks([])
            i += 1
            plt.subplot(10, 15, i)
            plt.imshow(np.array(grad_cov.cpu()), cmap='hot')
            plt.xticks([])
            plt.yticks([])
            i += 1
            if i > 150:
                break
        plt.show()

    def computer_score(self):

        for name in self.name_to_statistic:

            print(name)
            info = self.name_to_statistic[name]

            forward_mean = info.f_cls.sum_mean / info.f_cls.counter
            forward_cov = (info.f_cls.sum_covariance / info.f_cls.counter) - \
                torch.mm(forward_mean.view(-1, 1), forward_mean.view(1, -1))
            print('f: ', forward_cov.shape, torch.matrix_rank(forward_cov))

            # f_cov_inverse = forward_cov.inverse()
            # alpha = torch.reciprocal(torch.diag(f_cov_inverse))
            #
            # stack_op_for_weight = torch.mm(alpha.view(1, -1), f_cov_inverse)
            #
            # print(alpha, stack_op_for_weight)

            weight = info.module.weight.detach()
            print('module: ', weight.shape)

            grad_mean = info.b_cls.sum_mean / info.b_cls.counter
            grad_cov = (info.b_cls.sum_covariance / info.b_cls.counter) - \
                torch.mm(grad_mean.view(-1, 1), grad_mean.view(1, -1))
            eig_value, eig_vec = torch.eig(grad_cov, eigenvectors=True)

            adjust_matrix = torch.mm(torch.diag(torch.sqrt(eig_value[:, 0])), eig_vec.t()).to(torch.float)
            print('M: ', adjust_matrix.shape)

            adjusted_weight = torch.mm(adjust_matrix, torch.squeeze(weight))

