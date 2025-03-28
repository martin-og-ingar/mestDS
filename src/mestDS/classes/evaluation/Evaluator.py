from pathlib import Path


class Evaluator:
    model_path: str
    model_name: str
    train_command: str
    test_command: str
    is_github: bool
    commit: str

    def __init__(self, config):
        model_path = config.get["model_path"]
        self.train_command = config.get["train_command"]
        self.test_command = config.get["test_command"]

        if model_path.startswith("https://github.com"):
            dir_name = model_path.split("/")[-1].replace(".git", "")
            self.model_name = dir_name
            if "@" in model_path:
                self.model_path, self.commit = model_path.split("@")
            self.is_github = True
        else:
            self.model_name = Path(model_path).stem
            self.model_path = model_path
