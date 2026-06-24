"""Local aesthetic predictor wrappers used by :mod:`eigo`.

The implementations use the installed ``openai-clip`` package and keep model
weights in the same user cache used by the original predictor projects.
"""

from .laion_rank_image import LAIONAesthetic
from .laion_v2_rank_image import LAIONV2Aesthetic
from .simulacra_rank_image import SimulacraAesthetic

__all__ = ["LAIONAesthetic", "LAIONV2Aesthetic", "SimulacraAesthetic"]
