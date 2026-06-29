import torch
from torch import Tensor


def hard_loss(predict, target) -> Tensor:
    """
    Функция для лосса
    :param predict: Тенсор, созданный сеткой
    :param target:  Тенсор, который мы принимаем за правду
    :return:
    """
    prob = torch.sigmoid(predict)

    bce =  torch.nn.functional.binary_cross_entropy_with_logits(predict, target)

    uncertainty = 1 - torch.abs(prob * 2 - 1)
    uncertainty_penalty = uncertainty.mean()

    return bce + 0.5 * uncertainty_penalty