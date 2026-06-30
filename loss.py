import torch
from torch import nn
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


loss_function_table = {
    # Самописные реализации loss функции
    "hard_loss": hard_loss,

    # Реализации loss функции из pytorch
    "L1Loss": nn.L1Loss,
    "MSELoss": nn.MSELoss,
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "NLLLoss": nn.NLLLoss,
    "BCELoss": nn.BCELoss,
    "BCEWithLogitsLoss": nn.BCEWithLogitsLoss,
    "HuberLoss": nn.HuberLoss,
    "SmoothL1Loss": nn.SmoothL1Loss,
    "KLDivLoss": nn.KLDivLoss,
    "CTCLoss": nn.CTCLoss,
    "PoissonNLLLoss": nn.PoissonNLLLoss,
    "MarginRankingLoss": nn.MarginRankingLoss,
    "HingeEmbeddingLoss": nn.HingeEmbeddingLoss,
    "CosineEmbeddingLoss": nn.CosineEmbeddingLoss,
    "MultiMarginLoss": nn.MultiMarginLoss,
    "MultiLabelMarginLoss": nn.MultiLabelMarginLoss,
    "MultiLabelSoftMarginLoss": nn.MultiLabelSoftMarginLoss,
    "SoftMarginLoss": nn.SoftMarginLoss,
    "TripletMarginLoss": nn.TripletMarginLoss,
    "TripletMarginWithDistanceLoss": nn.TripletMarginWithDistanceLoss,
}
