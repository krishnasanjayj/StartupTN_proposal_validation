"""
Model evaluation module for comparing Base vs Fine-tuned Qwen models on proposal evaluation metrics.
"""

import json
from typing import List, Dict, Any
from src.llm.evaluator import ProposalEvaluator

class LLMEvaluator:
    def __init__(self, model_name_or_path: str = "Qwen/Qwen3-0.6B", device: str = None):
        self.evaluator = ProposalEvaluator(model_name_or_path=model_name_or_path, device=device)

    def evaluate_test_set(self, test_proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs evaluation over a list of test proposals and compiles performance metrics.
        """
        results = []
        valid_json_count = 0
        total = len(test_proposals)
        
        for idx, item in enumerate(test_proposals, 1):
            proposal_text = item.get("text", "")
            metadata = item.get("metadata", {})
            
            output = self.evaluator.evaluate(proposal_text, metadata=metadata)
            eval_dict = output["evaluation"]
            
            is_valid = not eval_dict.get("parse_error", False)
            if is_valid:
                valid_json_count += 1
                
            results.append({
                "proposal_id": item.get("proposal_id", f"TEST-{idx}"),
                "is_valid_json": is_valid,
                "recommendation": eval_dict.get("recommendation", "N/A"),
                "risk_count": len(eval_dict.get("risks", [])),
                "confidence_score": eval_dict.get("confidence_score", 0.0)
            })

        json_validity_rate = (valid_json_count / total) * 100 if total > 0 else 0.0
        
        return {
            "total_proposals": total,
            "json_validity_rate_pct": json_validity_rate,
            "detailed_results": results
        }
