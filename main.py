import segmentation_models_pytorch as smp
from PIL.Image import Image

from synthetic import DMSyntheticDataset
## Для отрисовки в картинку.
import torchvision.transforms as T

from train import DMTrainModel


def main():
    model = DMTrainModel()
    # model.start_training(20, 200)
    model.load("./test")
    # model.save("./test")
    # model.test()
    model.test_img("in.png")


if __name__ == "__main__":
    main()
