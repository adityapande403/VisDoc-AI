import pytest
import torch
import torch.nn.functional as F
from visdoc.quantization.fp16 import FP16Quantizer
from visdoc.quantization.int8 import INT8Quantizer
from visdoc.quantization.binary import BinaryQuantizer
from visdoc.evaluation.metrics import compute_quantization_error

def test_fp16_quantizer():
    torch.manual_seed(42)
    embs = F.normalize(torch.randn(10, 128), p=2, dim=-1)
    quantizer = FP16Quantizer()

    q_tensor, meta = quantizer.quantize(embs)
    assert q_tensor.dtype == torch.float16
    assert q_tensor.element_size() == 2

    recon = quantizer.dequantize(q_tensor, meta)
    err = compute_quantization_error(embs, recon)
    assert err["mse"] < 1e-6
    assert err["mean_cosine_similarity"] > 0.9999

def test_int8_quantizer():
    torch.manual_seed(42)
    embs = F.normalize(torch.randn(10, 128), p=2, dim=-1)
    quantizer = INT8Quantizer(mode="dequant")

    q_tensor, meta = quantizer.quantize(embs)
    assert q_tensor.dtype == torch.int8
    assert q_tensor.element_size() == 1

    recon = quantizer.dequantize(q_tensor, meta)
    err = compute_quantization_error(embs, recon)
    # High fidelity INT8 reconstruction
    assert err["mse"] < 0.001
    assert err["mean_cosine_similarity"] > 0.998

def test_binary_quantizer():
    torch.manual_seed(42)
    embs = F.normalize(torch.randn(10, 128), p=2, dim=-1)
    quantizer = BinaryQuantizer(embedding_dim=128)

    q_tensor, meta = quantizer.quantize(embs)
    assert q_tensor.dtype == torch.uint8
    assert q_tensor.shape == (10, 16) # 16 bytes per vector
    assert meta["bytes_per_vector"] == 16

    recon = quantizer.dequantize(q_tensor, meta)
    err = compute_quantization_error(embs, recon)
    # Binary quantization captures positive/negative sign quadrant
    assert err["mean_cosine_similarity"] > 0.70 # Theoretical random is 0.0, true binary sign preserves positive alignment
