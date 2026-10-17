"""Phoneme accuracy scoring. No model is loaded or downloaded on import."""
from .service import PronunciationScorer, configure_default, score_pronunciation

__version__ = "0.3.0"
__all__ = ["PronunciationScorer", "configure_default", "score_pronunciation"]
