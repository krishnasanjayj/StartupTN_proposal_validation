"""
Dataset Schema definition for Startup Proposal instruction tuning.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional
import json

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Text content of the message")

    @field_validator("role")

    def validate_role(cls, v):
        allowed = ["system", "user", "assistant"]
        if v not in allowed:
            raise ValueError(f"Role must be one of {allowed}, got '{v}'")
        return v

class ProposalEvaluationOutput(BaseModel):
    summary: str
    problem: str
    solution: str
    target_market: str
    business_model: str
    revenue_model: str
    strengths: List[str]
    weaknesses: List[str]
    risks: List[str]
    scalability: str
    recommendation: str
    confidence_score: float

class StartupProposalExample(BaseModel):
    proposal_id: str
    industry: str
    category: str
    is_synthetic: bool = True
    messages: List[ChatMessage]

    @field_validator("messages")

    def validate_messages(cls, messages):
        if len(messages) < 2:
            raise ValueError("Example must have at least 2 messages (user and assistant)")
            
        roles = [m.role for m in messages]
        if "user" not in roles or "assistant" not in roles:
            raise ValueError("Messages must contain at least 'user' and 'assistant' roles")
            
        # Validate that assistant content is valid JSON matching output schema
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        if assistant_msgs:
            last_assistant_content = assistant_msgs[-1].content
            try:
                parsed = json.loads(last_assistant_content)
                ProposalEvaluationOutput(**parsed)
            except Exception as e:
                raise ValueError(f"Assistant content is not valid ProposalEvaluationOutput JSON: {e}")
                
        return messages
