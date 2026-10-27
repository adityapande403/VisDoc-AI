import pytest
import torch
import torch.nn.functional as F
from visdoc.scoring.maxsim import exact_maxsim, compute_maxsim_scores, batched_maxsim_matrix
from visdoc.scoring.bitwise import pack_bits_tensor, unpack_bits_tensor, binary_maxsim

def test_exact_maxsim_hand_calculated():
    # 2 query tokens, 3 doc tokens, dim = 2
    # Q:
    # q0 = [1.0, 0.0]
    # q1 = [0.0, 1.0]
    # D:
    # d0 = [1.0, 0.0]
    # d1 = [0.0, -1.0]
    # d2 = [0.6, 0.8] (L2 norm = 1.0)
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = torch.tensor([[1.0, 0.0], [0.0, -1.0], [0.6, 0.8]])

    # Similarities:
    # q0 . d0 = 1.0, q0 . d1 = 0.0, q0 . d2 = 0.6 -> max for q0 is 1.0 (d0)
    # q1 . d0 = 0.0, q1 . d1 = -1.0, q1 . d2 = 0.8 -> max for q1 is 0.8 (d2)
    # Expected MaxSim = 1.0 + 0.8 = 1.8
    score = exact_maxsim(q, d)
    assert pytest.approx(score.item(), abs=1e-5) == 1.8

def test_exact_maxsim_with_mask():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    d = torch.tensor([[1.0, 0.0], [0.0, -1.0], [0.6, 0.8]])

    # Mask out second query token (q1): MaxSim should only be max for q0 -> 1.0
    q_mask = torch.tensor([True, False])
    score = exact_maxsim(q, d, query_mask=q_mask)
    assert pytest.approx(score.item(), abs=1e-5) == 1.0

    # Mask out d0: for q0, max between d1 (0.0) and d2 (0.6) is 0.6
    # for q1, max is d2 (0.8)
    # Expected sum = 0.6 + 0.8 = 1.4
    d_mask = torch.tensor([False, True, True])
    score2 = exact_maxsim(q, d, doc_mask=d_mask)
    assert pytest.approx(score2.item(), abs=1e-5) == 1.4

def test_batched_maxsim_equivalence():
    torch.manual_seed(42)
    queries = torch.randn(3, 4, 16)
    docs = torch.randn(5, 8, 16)

    matrix = batched_maxsim_matrix(queries, docs)
    assert matrix.shape == (3, 5)

    for i in range(3):
        for j in range(5):
            ref = exact_maxsim(queries[i], docs[j])
            assert pytest.approx(matrix[i, j].item(), abs=1e-4) == ref.item()

def test_bitwise_packing_and_binary_maxsim():
    torch.manual_seed(123)
    dim = 128
    bool_vec = torch.randn(5, dim) > 0
    packed = pack_bits_tensor(bool_vec)
    assert packed.shape == (5, dim // 8) # (5, 16)
    assert packed.dtype == torch.uint8

    unpacked = unpack_bits_tensor(packed)
    assert torch.equal(bool_vec, unpacked)

    # Identical query and doc in binary: Hamming distance 0 -> cosine similarity 1.0 per token
    q_packed = packed[:2]
    score = binary_maxsim(q_packed, q_packed, embedding_dim=dim)
    # 2 tokens * 1.0 = 2.0
    assert pytest.approx(score.item(), abs=1e-5) == 2.0
