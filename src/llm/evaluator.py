"""
Proposal Evaluator class wrapping Qwen inference and structured output parsing.
"""

import json
import re
from typing import Dict, Any, Optional, Tuple
from .loader import load_qwen_model
from .templates import SYSTEM_EVALUATION_PROMPT, format_proposal_prompt

class ProposalEvaluator:
    def __init__(self, model_name_or_path: str = "Qwen/Qwen3-0.6B", device: Optional[str] = None):
        self.model, self.tokenizer = load_qwen_model(model_name_or_path=model_name_or_path, device=device)

    def evaluate(self, proposal_text: str, metadata: Optional[Dict[str, Any]] = None, max_new_tokens: int = 768) -> Dict[str, Any]:
        """
        Evaluates a startup proposal and returns structured output along with thinking trace.
        """
        formatted_user_prompt = format_proposal_prompt(proposal_text, metadata)
        
        messages = [
            {"role": "system", "content": SYSTEM_EVALUATION_PROMPT},
            {"role": "user", "content": formatted_user_prompt}
        ]
        
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        
        inputs = self.tokenizer([input_text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0
        )
        
        output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
        
        # Parse Qwen thinking output
        thinking_content, content = self._extract_thinking_and_content(output_ids)
        parsed_json = self._parse_json_response(content)
        
        return {
            "evaluation": parsed_json,
            "raw_response": content,
            "thinking_process": thinking_content
        }

    def _extract_thinking_and_content(self, output_ids: list) -> Tuple[str, str]:
        """
        Splits output token IDs around Qwen's thinking end token (151668 for </think>).
        """
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
            
        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip()
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        return thinking_content, content

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Extracts and parses JSON object from model output text.
        """
        # Try finding JSON block enclosed in markdown ```json ... ```
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Fallback to finding raw brace match
            brace_match = re.search(r"\{.*\}", text, re.DOTALL)
            json_str = brace_match.group(0) if brace_match else text

        try:
            return json.loads(json_str)
        except Exception:
            return {
                "summary": text[:200] + "...",
                "problem": "Failed to parse structured JSON.",
                "solution": "N/A",
                "target_market": "N/A",
                "business_model": "N/A",
                "revenue_model": "N/A",
                "strengths": [],
                "weaknesses": [],
                "risks": ["Output formatting error from model."],
                "scalability": "N/A",
                "recommendation": "INSUFFICIENT_INFORMATION",
                "confidence_score": 0.0,
                "parse_error": True
            }
