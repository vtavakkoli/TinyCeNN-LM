import torch

from tinycenn_lm.pdelta2_features import (
    FeaturePDelta2Layer,
    PDelta2Core,
    dilated_sparse_attention,
)


def test_vectorized_preconditioner_matches_token_oracle():
    torch.manual_seed(3)
    core = PDelta2Core(4, 2, 8, feature_dim=12, chunk_size=5)
    kp = torch.randn(2, 2, 13, 12)
    kp = torch.nn.functional.normalize(kp, dim=-1)
    curvature = torch.rand(2, 2, 12) + 0.5
    a, ac = core.precondition_keys(kp, curvature.clone())
    b, bc = core.precondition_keys_reference(kp, curvature.clone())
    torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(ac, bc, rtol=2e-5, atol=2e-6)


def test_dilated_attention_is_causal():
    torch.manual_seed(4)
    q = torch.randn(1, 4, 20, 8)
    k = torch.randn(1, 2, 20, 8)
    v = torch.randn(1, 2, 20, 8)
    base = dilated_sparse_attention(q, k, v, [0, 1, 2, 4, 8], 2)
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 12:] += 50 * torch.randn_like(k2[:, :, 12:])
    v2[:, :, 12:] += 50 * torch.randn_like(v2[:, :, 12:])
    changed = dilated_sparse_attention(q, k2, v2, [0, 1, 2, 4, 8], 2)
    torch.testing.assert_close(base[:, :, :12], changed[:, :, :12], rtol=0, atol=1e-6)


def test_feature_layer_streaming_matches_full_prefix():
    torch.manual_seed(5)
    layer = FeaturePDelta2Layer(
        4, 2, 8, feature_dim=16, retrieval="dilated",
        offsets=[0, 1, 2, 4, 8], dual_timescale=True, chunk_size=4,
    )
    q = torch.randn(1, 4, 16, 8)
    k = torch.randn(1, 2, 16, 8)
    v = torch.randn(1, 2, 16, 8)
    full = layer(q, k, v)
    first, state = layer(q[:, :, :8], k[:, :, :8], v[:, :, :8], return_state=True)
    second, _ = layer(q[:, :, 8:], k[:, :, 8:], v[:, :, 8:], state=state, return_state=True)
    torch.testing.assert_close(full, torch.cat((first, second), dim=2), rtol=2e-5, atol=2e-5)


def test_config_reconstructs_checkpoint_shape():
    layer = FeaturePDelta2Layer(
        6, 3, 8, feature_dim=18, retrieval="dense", window=7,
        dual_timescale=True, chunk_size=4,
    )
    clone = FeaturePDelta2Layer(**layer.config)
    clone.load_state_dict(layer.state_dict())
    assert clone.config == layer.config
    assert clone.recurrent_state_bytes() == layer.recurrent_state_bytes()


def test_sparse_retrieval_uses_fewer_pairs_than_dense_window():
    dense = FeaturePDelta2Layer(4, 2, 8, feature_dim=16, retrieval="dense", window=32)
    sparse = FeaturePDelta2Layer(
        4, 2, 8, feature_dim=16, retrieval="dilated",
        offsets=[0, 1, 2, 4, 8, 16, 32],
    )
    assert sparse.retrieval_pairs_per_token() == 7
    assert dense.retrieval_pairs_per_token() == 32
    assert sparse.retrieval_pairs_per_token() < dense.retrieval_pairs_per_token()
