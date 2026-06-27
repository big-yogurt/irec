import segmentation_models_pytorch as smp
from PIL.Image import Image

from synthetic import DMSyntheticDataset
## Для отрисовки в картинку.
import torchvision.transforms as T

from train import DMTrainModel


def main():
    # model = smp.UnetPlusPlus(
    #     # TODO: Нужно поиграться с настройками
    #     encoder_name="efficientnet-b3",
    #     encoder_weights=None,
    #     in_channels=3,
    #     classes=1,
    #     activation=None,
    # )

    model = DMTrainModel()
    model.start_training()
    model.test()
    ## Генерим рандомный экземпляр и сохраняем его.
    # synth = DMSyntheticDataset()
    # data = synth[0]
    #
    # input: Image = T.ToPILImage()(data["input_tensor"])
    # target: Image = T.ToPILImage()(data["target_tensor"])
    #
    # # display the PIL image
    # input.show("in")
    # target.show("out")

if __name__ == "__main__":
    main()
