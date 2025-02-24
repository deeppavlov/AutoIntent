from .abstract import AbstractEvolution
from .base import EvolutionChatTemplate
from .concrete import ConcreteEvolution
from .formal import FormalEvolution
from .funny import FunnyEvolution
from .goofy import GoofyEvolution
from .informal import InformalEvolution
from .reasoning import ReasoningEvolution

EVOLUTION_NAMES = [evolution.name for evolution in EvolutionChatTemplate.__subclasses__()]

EVOLUTION_MAPPING = {evolution.name: evolution() for evolution in EvolutionChatTemplate.__subclasses__()}

__all__ = [
    "EVOLUTION_MAPPING",
    "EVOLUTION_NAMES",
    "AbstractEvolution",
    "ConcreteEvolution",
    "EvolutionChatTemplate",
    "FormalEvolution",
    "FunnyEvolution",
    "GoofyEvolution",
    "InformalEvolution",
    "ReasoningEvolution",
]
