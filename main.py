import os
import sys
import argparse
from datetime import datetime

import cli


def main():
    cli_flags = cli.parse_cli()
    match cli_flags.command:
        case "run":
            command_run(cli_flags)
        case "train":
            command_train(cli_flags)
        case "test":
            command_test(cli_flags)
        case "syn_test":
            command_syn_test(cli_flags)
        case "gen_ds":
            command_gen_ds(cli_flags)


def command_gen_ds(cli_flags: argparse.Namespace):
    ds_path = cli_flags.path + "/" + datetime.now().strftime("%d%m%y_%H%M%S")
    imgs_path = ds_path + "/imgs"
    masks_path = ds_path + "/masks"
    os.makedirs(imgs_path)
    os.makedirs(masks_path)
    import cv2
    import imggen
    for i in range(cli_flags.len):
        img, mask = imggen.gen_random_img_pair()
        cv2.imwrite(imgs_path + "/" + str(i) + ".png", img)
        cv2.imwrite(masks_path + "/" + str(i) + ".png", mask)


def command_run(cli_flags: argparse.Namespace):
    # TODO: реализовать запуск нейросети и получение данных через сеть
    ...


def command_train(cli_flags: argparse.Namespace):
    if not os.path.isdir(cli_flags.train_ds_path):
        print(f"Файл датасета '{cli_flags.train_ds_path}' не найден")
        sys.exit(1)
    if not os.path.isdir(cli_flags.val_ds_path):
        print(f"Файл датасета '{cli_flags.val_ds_path}' не найден")
        sys.exit(1)
    import train
    import synthetic
    model = train.DMTrainModel()
    if cli_flags.load_nn is not None:
        model.load(cli_flags.load_nn)
    model.start_training(
        cli_flags.epoch,
        cli_flags.batch,
        cli_flags.num_workers,
        synthetic.DMPreloadedDataset(cli_flags.train_ds_path),
        synthetic.DMPreloadedDataset(cli_flags.val_ds_path),
    )
    model.save(cli_flags.save_nn + "/" +
        datetime.now().strftime("%d%m%y_%H%M%S")
    )


def command_test(cli_flags: argparse.Namespace):
    import train
    model = train.DMTrainModel()
    model.load(cli_flags.load_nn)
    model.test_img(cli_flags.path_to_image)


def command_syn_test(cli_flags: argparse.Namespace):
    import train
    model = train.DMTrainModel()
    model.load(cli_flags.load_nn)
    model.test()


if __name__ == "__main__":
    main()
