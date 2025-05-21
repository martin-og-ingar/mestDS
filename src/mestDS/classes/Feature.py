from dataclasses import dataclass
from typing import Any


class Variable:
    def __init__(self):
        self.name: str = ""
        self.function: str = ""
        self.params: dict[str, Any] = {}
