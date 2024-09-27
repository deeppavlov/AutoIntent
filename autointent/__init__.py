import sys

from .context import Context  # noqa: F401

__import__("pysqlite3")

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
