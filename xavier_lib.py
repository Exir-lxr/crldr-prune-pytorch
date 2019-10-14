import torch
import numpy as np


class InfoStruct(object):

    def __init__(self, module, pre_f_cls, f_cls, b_cls):
        self.module = module
        self.pre_f_cls = pre_f_cls
        self.f_cls = f_cls
        self.b_cls = b_cls

        self.stacked_ops = None
        self.score = None
        self.arg_sort = None
        # equal 0 where variance of an activate is 0
        self.zero_variance_mask = None

    def record(self, ops, score, zero_variance_mask):
        self.stacked_ops = ops
        self.score = score
        self.arg_sort = torch.argsort(score)
        self.zero_variance_mask = zero_variance_mask

    def last_value(self):
        for i in range(list(self.score.shape)[0]):
            index = int(self.arg_sort[i])
            if int(self.zero_variance_mask[index]) != 0:
                return index, float(self.score[index])

    def prune_a_channel(self, index):
        pass


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
            elif self.dim == 2:
                samples = inputs[0]
            compute_statistic_and_update(samples, self.sum_mean, self.sum_covariance, self.counter)


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
            if self.dim == 4:
                samples = grad_output[0].permute(0, 2, 3, 1).contiguous().view(-1, channel_num)
            elif self.dim == 2:
                samples = grad_output[0]
            compute_statistic_and_update(samples, self.sum_mean, self.sum_covariance, self.counter)


class PreForwardHook(object):

    def __init__(self, name, dim=4):
        self.name = name
        self.dim=dim
        self.mask = None
        self.base = None

    def __call__(self, module, inputs):
        channel_num = list(inputs[0].shape)[1]
        if self.mask is None:
            self.mask = torch.nn.Parameter(torch.ones(channel_num), requires_grad=False).cuda()
            self.base = torch.nn.Parameter(torch.zeros(channel_num), requires_grad=False).cuda()
        if self.dim == 4:
            modified = torch.mul(inputs[0].permute([0, 2, 3, 1]), self.mask) + self.base
            return tuple(modified.permute([0, 3, 1, 2]), )
        elif self.dim == 2:
            return tuple(torch.mul(inputs, self.mask) + self.base, )

    def update_mask_base(self, new_mask, new_base):
        self.mask.data = new_mask
        self.base.data = new_base


class StatisticManager(object):

    def __init__(self):

        self.name_to_statistic = {}
        self.bn_name = {}

    def __call__(self, model):

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, torch.nn.Conv2d):
                if sub_module.kernel_size[0] == 1:
                    pre_hook_cls = PreForwardHook(name)
                    hook_cls = ForwardStatisticHook(name)
                    back_hook_cls = BackwardStatisticHook(name)
                    sub_module.register_forward_pre_hook(pre_hook_cls)
                    sub_module.register_forward_hook(hook_cls)
                    sub_module.register_backward_hook(back_hook_cls)
                    self.name_to_statistic[name] = InfoStruct(sub_module, pre_hook_cls, hook_cls, back_hook_cls)
                    print('conv', name)
            elif isinstance(sub_module, torch.nn.Linear):
                pre_hook_cls = PreForwardHook(name, dim=2)
                hook_cls = ForwardStatisticHook(name, dim=2)
                back_hook_cls = BackwardStatisticHook(name, dim=2)
                sub_module.register_forward_hook(hook_cls)
                sub_module.register_backward_hook(back_hook_cls)
                self.name_to_statistic[name] = InfoStruct(sub_module, pre_hook_cls, hook_cls, back_hook_cls)
                print('conv', name)
            elif isinstance(sub_module, torch.nn.BatchNorm1d) or isinstance(sub_module, torch.nn.BatchNorm2d):
                self.bn_name[name] = sub_module
                print('bn', name)

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

        with torch.no_grad():

            for name in self.name_to_statistic:

                info = self.name_to_statistic[name]

                # compute forward covariance
                forward_mean = info.f_cls.sum_mean / info.f_cls.counter
                forward_cov = (info.f_cls.sum_covariance / info.f_cls.counter) - \
                    torch.mm(forward_mean.view(-1, 1), forward_mean.view(1, -1))
                # print('f: ', forward_cov.shape, torch.matrix_rank(forward_cov))

                channel_num = list(forward_cov.shape)[0]

                # equal 0 where variance of an activate is 0
                zero_variance_mask = torch.sign(torch.diag(forward_cov))

                forward_cov += torch.diag(torch.nn.Parameter(torch.ones(channel_num, dtype=torch.double)).cuda()-\
                                          zero_variance_mask)

                f_cov_inverse = forward_cov.inverse().to(torch.float)
                alpha = torch.reciprocal(torch.diag(f_cov_inverse))

                stack_op_for_weight = (f_cov_inverse.t() * alpha.view(1, -1)).t()

                # print(alpha)

                weight = info.module.weight.detach()
                # print('module: ', weight.shape)

                grad_mean = info.b_cls.sum_mean / info.b_cls.counter
                grad_cov = (info.b_cls.sum_covariance / info.b_cls.counter) - \
                    torch.mm(grad_mean.view(-1, 1), grad_mean.view(1, -1))
                eig_value, eig_vec = torch.eig(grad_cov, eigenvectors=True)

                adjust_matrix = torch.mm(torch.diag(torch.sqrt(eig_value[:, 0])), eig_vec.t()).to(torch.float)
                # print('M: ', adjust_matrix.shape)

                adjusted_weight = torch.abs(torch.mm(adjust_matrix, torch.squeeze(weight)))
                score = torch.mm(adjusted_weight, alpha.view([-1, 1])).view([-1])
                # print

                info.record(stack_op_for_weight, score, zero_variance_mask)
