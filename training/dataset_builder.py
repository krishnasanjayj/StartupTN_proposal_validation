"""
Dataset Cleaning, Anonymization, Validation, and Train/Val/Test Split Utilities.
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from .dataset_schema import StartupProposalExample

class ProposalDatasetBuilder:
    def __init__(self, raw_data_path: Optional[str] = None):
        self.raw_data_path = Path(raw_data_path) if raw_data_path else None

    @staticmethod
    def anonymize_text(text: str) -> str:
        """
        Removes sensitive personal information like emails, phone numbers, and Aadhaar/PAN IDs.
        """
        # Email pattern
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[ANONYMIZED_EMAIL]', text)
        # Indian phone number pattern
        text = re.sub(r'(?:\+91[\-\s]?)?[6-9]\d{9}', '[ANONYMIZED_PHONE]', text)
        # PAN card format (5 alpha, 4 numeric, 1 alpha)
        text = re.sub(r'[A-Z]{5}[0-9]{4}[A-Z]{1}', '[ANONYMIZED_PAN]', text)
        # Aadhaar format (12 digits)
        text = re.sub(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b', '[ANONYMIZED_AADHAAR]', text)
        return text

    def load_and_validate_jsonl(self, file_path: str) -> Tuple[List[StartupProposalExample], List[str]]:
        """
        Loads a JSONL dataset file, anonymizes text, and validates each record against Pydantic schema.
        """
        valid_examples = []
        errors = []
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # Anonymize user messages
                    for msg in data.get("messages", []):
                        msg["content"] = self.anonymize_text(msg["content"])
                        
                    example = StartupProposalExample(**data)
                    valid_examples.append(example)
                except Exception as e:
                    errors.append(f"Line {line_no}: {e}")
                    
        return valid_examples, errors

    def remove_duplicates(self, examples: List[StartupProposalExample]) -> List[StartupProposalExample]:
        """
        Deduplicates dataset based on exact proposal user prompt content.
        """
        seen = set()
        deduped = []
        for ex in examples:
            user_text = next((m.content for m in ex.messages if m.role == "user"), "")
            cleaned = re.sub(r'\s+', ' ', user_text).strip().lower()
            if cleaned not in seen:
                seen.add(cleaned)
                deduped.append(ex)
        return deduped

    def split_dataset(
        self, 
        examples: List[StartupProposalExample], 
        train_ratio: float = 0.8, 
        val_ratio: float = 0.1, 
        seed: int = 42
    ) -> Tuple[List[StartupProposalExample], List[StartupProposalExample], List[StartupProposalExample]]:
        """
        Splits dataset into Train, Validation, and Test sets safely without content leakage.
        """
        random.seed(seed)
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_set = shuffled[:n_train]
        val_set = shuffled[n_train:n_train + n_val]
        test_set = shuffled[n_train + n_val:]
        
        return train_set, val_set, test_set

    def export_jsonl(self, examples: List[StartupProposalExample], output_path: str):
        """
        Exports a list of StartupProposalExample objects to JSONL file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex.model_dump(), ensure_ascii=False) + "\n")
