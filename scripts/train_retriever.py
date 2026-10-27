import sys
from pathlib import Path
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visdoc.training.trainer import ColBERTVisualTrainer
from visdoc.utils.logging import logger

def main():
    logger.info("Initializing ColBERTVisualTrainer smoke test harness...")
    dummy_model = nn.Linear(128, 128)
    trainer = ColBERTVisualTrainer(
        model=dummy_model,
        output_dir="d:/Aditya/hf6/checkpoints/finetuned",
        learning_rate=1e-4,
    )

    smoke_results = trainer.run_smoke_test(embedding_dim=128)
    logger.info(f"Smoke test completed: {smoke_results}")

if __name__ == "__main__":
    main()
