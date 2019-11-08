# -*- coding: utf-8 -*-
# Source code from https://github.com/dizam92/pyTorchReg/
__author__ = 'maoss2'


class _Regularizer(object):
    """
    Parent class of Regularizers
    """

    def __init__(self, model):
        super(_Regularizer, self).__init__()
        self.model = model

    def regularized_param(self, param_weights, reg_loss_function):
        raise NotImplementedError

    def regularized_all_param(self, reg_loss_function):
        raise NotImplementedError


class L1Regularizer(_Regularizer):
    """
    L1 regularized loss
    """

    def __init__(self, model, lambda_reg=0.01, exclusions=None):
        super(L1Regularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg
        self.exclusions = exclusions if exclusions is not None else []

    def regularized_param(self, param_weights, reg_loss_function):
        reg_loss_function += self.lambda_reg * L1Regularizer.__add_l1(var=param_weights)
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if True not in [exclusion in model_param_name for
                            exclusion in self.exclusions]:
                reg_loss_function += self.lambda_reg * L1Regularizer.__add_l1(
                    var=model_param_value)
        return reg_loss_function

    @staticmethod
    def __add_l1(var):
        return var.abs().sum()


class L2Regularizer(_Regularizer):
    """
       L2 regularized loss
    """

    def __init__(self, model, lambda_reg=0.01, exclusions=None):
        super(L2Regularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg
        self.exclusions = exclusions if exclusions is not None else []

    def regularized_param(self, param_weights, reg_loss_function):
        reg_loss_function += self.lambda_reg / 2 * L2Regularizer.__add_l2(var=param_weights)
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):

        for model_param_name, model_param_value in self.model.named_parameters():
            if True not in [exclusion in model_param_name for
                            exclusion in self.exclusions]:
                print(model_param_name)
                reg_loss_function += self.lambda_reg / 2 * L2Regularizer.__add_l2(
                    var=model_param_value)
        return reg_loss_function

    @staticmethod
    def __add_l2(var):
        return var.pow(2).sum()


regularizers = {
    'L1': L1Regularizer,
    'L2': L2Regularizer
}
