import segmentation_models_pytorch as smp

def main():
    model = smp.UnetPlusPlus(
        # TODO: Нужно поиграться с настройками
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )

if __name__ == "__main__":
    main()
