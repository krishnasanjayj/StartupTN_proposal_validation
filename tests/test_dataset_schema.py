"""
Unit test for dataset schema validation and sample dataset integrity.
"""

import pytest
from training.dataset_builder import ProposalDatasetBuilder

def test_sample_dataset_validation():
    builder = ProposalDatasetBuilder()
    file_path = "data/processed/startup_proposals_sample.jsonl"
    
    valid_examples, errors = builder.load_and_validate_jsonl(file_path)
    
    assert len(errors) == 0, f"Dataset schema validation failed with errors: {errors}"
    assert len(valid_examples) >= 4
    
    # Test deduplication
    deduped = builder.remove_duplicates(valid_examples)
    assert len(deduped) == len(valid_examples)
    
    # Test dataset splitting
    train, val, test = builder.split_dataset(valid_examples, train_ratio=0.5, val_ratio=0.25, seed=42)
    assert len(train) + len(val) + len(test) == len(valid_examples)
