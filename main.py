import argparse

import cli


def main():
    cli_flags = cli.parse_cli()
    match cli_flags.command:
        case "run":
            command_run(cli_flags)
        case "syn_train":
            command_syn_train(cli_flags)
        case "train":
            command_train(cli_flags)
        case "test":
            command_test(cli_flags)
        case "syn_test":
            command_syn_test(cli_flags)


def command_run(cli_flags: argparse.Namespace):
    # TODO: реализовать запуск нейросети и получение данных через сеть
    ...


def command_syn_train(cli_flags: argparse.Namespace):
    import train
    model = train.DMTrainModel()
    model.start_training(cli_flags.epoch, cli_flags.dataset_len)
    model.save(cli_flags.save_nn)


def command_train(cli_flags: argparse.Namespace):
    ...


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
