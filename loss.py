import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

def hard_loss(predict, target) -> Tensor:
    """
    Функция для лосса
    :param predict: Тенсор, созданный сеткой
    :param target:  Тенсор, который мы принимаем за правду
    :return:
    """
    prob = torch.sigmoid(predict)

    # 1. BCE — базовая классификация пикселей
    bce = F.binary_cross_entropy_with_logits(predict, target)

    # 2. Dice loss — штраф за нечёткое перекрытие регионов,
    # естественно продвигает чёткие границы
    smooth = 1e-6
    p_flat = prob.view(-1)
    t_flat = target.view(-1)
    intersection = (p_flat * t_flat).sum()
    dice = 1 - (2 * intersection + smooth) / (p_flat.sum() + t_flat.sum() + smooth)

    # 3. Uncertainty — штраф за пиксели застрявшие у 0.5
    uncertainty = (prob * (1 - prob)).mean() * 4  # *4 нормализует в диапазон 0..1

    # 4. Sharpness — штраф за "мягкие" переходы между соседними пикселями
    # g*(1-g) максимально при g=0.5, равно 0 при g=0 или g=1
    # то есть наказываем промежуточные переходы, а не резкие или плоские
    dx = (prob[:, :, :, 1:] - prob[:, :, :, :-1]).abs()  # горизонтальный градиент
    dy = (prob[:, :, 1:, :] - prob[:, :, :-1, :]).abs()  # вертикальный градиент
    grad = torch.cat([dx.reshape(-1), dy.reshape(-1)])
    sharpness_penalty = (grad * (1 - grad)).mean()

    return bce + dice + uncertainty + 0.3 * sharpness_penalty


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
