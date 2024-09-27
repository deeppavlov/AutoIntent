from autointent.modules.base import Module


class RetrievalModule(Module):
    def __init__(self, k: int):
        self.k = k
