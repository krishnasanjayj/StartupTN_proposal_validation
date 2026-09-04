"""
Qwen Model Loader for local CPU inference and Cloud GPU evaluation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, Optional

def load_qwen_model(
    model_name_or_path: str = "Qwen/Qwen3-0.6B",
    device: Optional[str] = None,
    adapter_path: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads Qwen model and tokenizer. Defaults to CPU if no GPU is available.
    Supports optional loading of fine-tuned PEFT LoRA adapters.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"[LLM Loader] Loading model '{model_name_or_path}' on device: '{device}'...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True
    )
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32 if device == "cpu" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True
    )
    
    if device == "cpu":
        model = model.to("cpu")
        
    if adapter_path:
        from peft import PeftModel
        print(f"[LLM Loader] Loading LoRA adapter weights from '{adapter_path}'...")
        model = PeftModel.from_pretrained(model, adapter_path)
        
    model.eval()
    print("[LLM Loader] Model loaded successfully.")
    return model, tokenizer
