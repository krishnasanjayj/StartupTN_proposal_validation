"""
StartupTN LLM Submodule for Proposal Evaluation.
"""
from .loader import load_qwen_model
from .evaluator import ProposalEvaluator
from .templates import SYSTEM_EVALUATION_PROMPT

__all__ = ["load_qwen_model", "ProposalEvaluator", "SYSTEM_EVALUATION_PROMPT"]
