import pytest
import torch.nn as nn
from visdoc.training.trainer import ColBERTVisualTrainer

def test_training_smoke_test(tmp_path):
    dummy_model = nn.Linear(128, 128)
    trainer = ColBERTVisualTrainer(
        model=dummy_model,
        output_dir=tmp_path,
        learning_rate=1e-4,
        device="cpu"
    )

    result = trainer.run_smoke_test(embedding_dim=128)
    assert result["status"] == "PASSED"
    assert result["loss"] > 0.0
    assert len(result["gradient_norms"]) > 0
