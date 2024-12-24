import json
import os
import string
from typing import Any


class SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return kwargs.get(key, "{" + key + "}")
        return super().get_value(key, args, kwargs)

    def parse(self, format_string):
        try:
            return super().parse(format_string)
        except ValueError:
            return [(format_string, None, None, None)]


def safe_format(format_string, *args, **kwargs):
    formatter = SafeFormatter()
    return formatter.format(format_string, *args, **kwargs)


def read_json_dataset(file_path: os.PathLike):
    with file_path.open("r") as file:
        return json.load(file)


def save_json_dataset(file_path: os.PathLike, intents: list[dict[str, Any]]):
    dirname = file_path.parent
    if not dirname.exists():
        dirname.makedir(dirname, parents=True)
    with file_path.open("w") as file:
        json.dump(intents, file, indent=4, ensure_ascii=False)
