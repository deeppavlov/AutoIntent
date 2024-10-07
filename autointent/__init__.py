import sys

from .context import Context

__import__("pysqlite3")

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

__all__ = ["Context"]
