from autointent.modules.base import Module


class RetrievalModule(Module):
    def __init__(self, k: int) -> None:
        self.k = k
