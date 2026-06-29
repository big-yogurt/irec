import segmentation_models_pytorch as smp
from PIL.Image import Image

from synthetic import DMSyntheticDataset
## Для отрисовки в картинку.
import torchvision.transforms as T

from train import DMTrainModel


def main():
    model = DMTrainModel()
    model.start_training(50, 250)
    #model.load("./test_hard_loss")
    model.save("./test_hard_loss")
    #model.test()
    # model.test_img("out.png")


if __name__ == "__main__":
    main()
