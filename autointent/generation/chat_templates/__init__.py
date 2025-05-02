"""Chat templates used throughout :py:mod:`autointent.generation` module."""

from ._abstract import AbstractEvolution
from ._base_evolver import EvolutionChatTemplate
from ._base_synthesizer import BaseSynthesizerTemplate
from ._concrete import ConcreteEvolution
from ._evolution_templates_schemas import Message, Role
from ._formal import FormalEvolution
from ._funny import FunnyEvolution
from ._goofy import GoofyEvolution
from ._informal import InformalEvolution
from ._intent_descriptions import PromptDescription
from ._reasoning import ReasoningEvolution
from ._synthesizer_en import EnglishSynthesizerTemplate
from ._synthesizer_ru import RussianSynthesizerTemplate

EVOLUTION_NAMES = [evolution.name for evolution in EvolutionChatTemplate.__subclasses__()]

EVOLUTION_MAPPING = {evolution.name: evolution() for evolution in EvolutionChatTemplate.__subclasses__()}

__all__ = [
    "EVOLUTION_MAPPING",
    "EVOLUTION_NAMES",
    "AbstractEvolution",
    "BaseSynthesizerTemplate",
    "ConcreteEvolution",
    "EnglishSynthesizerTemplate",
    "EvolutionChatTemplate",
    "FormalEvolution",
    "FunnyEvolution",
    "GoofyEvolution",
    "InformalEvolution",
    "Message",
    "PromptDescription",
    "ReasoningEvolution",
    "Role",
    "RussianSynthesizerTemplate",
]
