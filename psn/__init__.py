"""Phase Siren Network (PSN)

Non-gradient neural computation via hash-constructed specialized experts
and Kuramoto phase dynamics.
"""

from psn.model import PSN, HashDistributor
from psn.data import load_corpus

__all__ = ["PSN", "HashDistributor", "load_corpus"]
__version__ = "0.2.0"
