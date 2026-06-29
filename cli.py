import argparse


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="irec",
        description="Сервис улучшения качества маркировки datamatrix"
    )
    _setup_parser(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Команда 'run'
    run_parser = subparsers.add_parser("run", help="Запуск сервиса")
    _setup_run_parser(run_parser)

    # Команда 'syn_train'
    syn_train_parser = subparsers.add_parser("syn_train",
        help="Обучение модели на синтетических данных"
    )
    _setup_syn_train_parser(syn_train_parser)

    # Команда 'train'
    train_parser = subparsers.add_parser("train",
        help="Обучение модели на готовых данных"
    )
    _setup_train_parser(train_parser)

    # Команда 'test'
    test_parser = subparsers.add_parser("test",
        help="Тестирование модели"
    )
    _setup_test_parser(test_parser)

    # Команда 'syn_test'
    syn_test_parser = subparsers.add_parser("syn_test",
        help="Тестирование модели на синтетических данных"
    )
    _setup_syn_test_parser(syn_test_parser)


    return parser.parse_args()


def _setup_syn_test_parser(syn_test_parser: argparse.ArgumentParser):
    syn_test_parser.add_argument("--input_path", type=str, default="tests/in",
        help="Путь к папке, куда будут сохранены данные со входа нейросети"
    )
    syn_test_parser.add_argument("--output_path", type=str, default="tests/out",
        help="Путь к папке, куда будут сохранены результаты работы нейросети"
    )
    syn_test_parser.add_argument("--pair_count", type=int, default=10,
        help="Сколько пар (изображение, маска) будет сгенерировано"
    )


def _setup_test_parser(test_parser: argparse.ArgumentParser):
    test_parser.add_argument("path_to_image", type=str,
        help="Путь к изображению"
    )


def _setup_parser(parser: argparse.ArgumentParser):
    """
    Настройка флагов для всех команд
    """
    parser.add_argument("--save_nn", type=str, default="nn",
        help="Путь сохранения нейросети")
    parser.add_argument("--load_nn", type=str, default="nn",
        help="Путь загрузки нейросети")


def _setup_run_parser(run_parser: argparse.ArgumentParser):
    """
    Настройка флагов для команды 'run'
    """
    ...


def _setup_syn_train_parser(syn_train_parser: argparse.ArgumentParser):
    """
    Настройка флагов для команды 'syn_train'
    """
    syn_train_parser.add_argument("--epoch", type=int, help="Количество эпох",
        required=True)
    syn_train_parser.add_argument("--dataset_len", type=int,
        help="Количество пар изображений в датасете", required=True
    )


def _setup_train_parser(train_parser: argparse.ArgumentParser):
    """
    Настройка флагов для команды 'train'
    """
    train_parser.add_argument("--epoch", type=int, help="Количество эпох")
    train_parser.add_argument("--path", type=str, help="Путь к датасету")
