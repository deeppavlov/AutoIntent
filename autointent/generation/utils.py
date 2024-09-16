import importlib.resources as ires
import string


class SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return kwargs.get(key, "{" + key + "}")
        else:
            return super().get_value(key, args, kwargs)

    def parse(self, format_string):
        try:
            return super().parse(format_string)
        except ValueError:
            return [(format_string, None, None, None)]


def safe_format(format_string, *args, **kwargs):
    formatter = SafeFormatter()
    return formatter.format(format_string, *args, **kwargs)


def load_prompt(file_name: str):
    with ires.files("autointent.generation.prompts").joinpath(file_name).open() as file:
        return file.read()
