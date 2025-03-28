import subprocess


class CustomEvaluator:
    train_command: str
    test_command: str

    def __init__(self, config):
        self.train_command = config.get("train_command")
        self.test_command = config.get("test_command")

    def evaluate(self, simulation):
        train_command = [
            self.train_command[0]["language"],
            self.train_command[1]["file"],
        ]
        train_args = convert_args(self.train_command[2]["args"], simulation)

        train_command += train_args

        subprocess.run(train_command, check=True)

        test_command = [
            self.test_command[0]["language"],
            self.test_command[1]["file"],
        ]
        test_args = convert_args(self.test_command[2]["args"], simulation)

        test_command += test_args

        subprocess.run(test_command, check=True)


def convert_args(args, simulation):
    converted_args = []
    for arg in args:
        match arg:
            case "train_set":
                converted_args.append(simulation.test_set_x_path)
            case "train_set_x":
                converted_args.append(simulation.test_set_x_path)
            case "train_set_y":
                converted_args.append(simulation.test_set_y_path)
            case "test_set":
                converted_args.append(simulation.test_set_x_path)
            case "test_set_x":
                converted_args.append(simulation.test_set_x_path)
            case "test_set_y":
                converted_args.append(simulation.test_set_y_path)
            case _:
                converted_args.append(arg)

    return converted_args
