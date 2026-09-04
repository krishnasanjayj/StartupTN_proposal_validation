"""
Unit test for Qwen3-0.6B local CPU model loading and proposal evaluation inference.
"""

import pytest
from src.llm.evaluator import ProposalEvaluator

def test_qwen_cpu_inference():
    evaluator = ProposalEvaluator(model_name_or_path="Qwen/Qwen3-0.6B", device="cpu")
    
    sample_proposal = (
        "AgriTech Startup proposal: Developing smart automated pest traps using IoT sensors "
        "and computer vision for cotton farmers in Salem district, Tamil Nadu."
    )
    
    result = evaluator.evaluate(sample_proposal, max_new_tokens=256)
    
    assert "evaluation" in result
    assert "thinking_process" in result
    assert "raw_response" in result
    
    eval_data = result["evaluation"]
    assert isinstance(eval_data, dict)
    assert "summary" in eval_data
    assert "recommendation" in eval_data
