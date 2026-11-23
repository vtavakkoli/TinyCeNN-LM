"""Small deterministic unit tests for the causal No-CeNN Qwen3.5 mixer."""
import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from tinycenn_lm.qwen35_window_anchor import causal_window_anchor
def test_no_future_leakage_and_shapes():
    torch.manual_seed(7)
    q = torch.randn(1, 2, 12, 8)
    k = torch.randn(1, 2, 12, 8)
    v = torch.randn(1, 2, 12, 8)
    a = causal_window_anchor(q,k,v,None,window=3,anchors=1,chunk_size=4)
    v[:, :, 10:, :] += 1000.
    b = causal_window_anchor(q,k,v,None,window=3,anchors=1,chunk_size=4)
    assert a.shape == (1,2,12,8)
    torch.testing.assert_close(a[:,:,:10],b[:,:,:10],atol=1e-5,rtol=1e-5)

def test_anchors_preserve_prefix_tokens_outside_window():
    q=torch.ones((1,1,8,4))
    k=torch.zeros((1,1,8,4))
    v=torch.zeros((1,1,8,4))
    v[:,:,0,:]=100.
    with_anchor=causal_window_anchor(q,k,v,None,window=2,anchors=1,chunk_size=3)
    no_anchor=causal_window_anchor(q,k,v,None,window=2,anchors=0,chunk_size=3)
    assert with_anchor[0,0,7].mean()>20
    assert no_anchor[0,0,7].abs().max()<1e-6

def test_padding_is_ignored():
    q=torch.ones((1,1,5,4))
    k=torch.zeros((1,1,5,4))
    v=torch.ones((1,1,5,4))
    v[:,:,4,:]=9999
    valid=torch.tensor([[1,1,1,1,0]])
    result=causal_window_anchor(q,k,v,valid,window=5,anchors=1,chunk_size=2)
    assert torch.all(result[:,:,:4]<10)


def test_tiny_qwen_all24_full_forward_cpu():
    # No 1.75 GB model download required: initialize a tiny 24-block
    # Qwen3.5 text backbone, install all replacements, run causal logits.
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from tinycenn_lm.qwen35_window_anchor import (
        install_all24, WindowAnchorDelta, WindowAnchorFull)
    cfg=Qwen3_5TextConfig(
        vocab_size=128,hidden_size=64,intermediate_size=128,
        num_hidden_layers=24,num_attention_heads=2,num_key_value_heads=1,
        head_dim=32,linear_num_key_heads=2,linear_num_value_heads=2,
        linear_key_head_dim=32,linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        layer_types=["full_attention" if i%4==3 else "linear_attention"
                     for i in range(24)],
        rope_parameters={"rope_type":"default","rope_theta":10000.0,
                         "partial_rotary_factor":0.25,
                         "mrope_interleaved":True,
                         "mrope_section":[2,1,1]},
        use_cache=False)
    torch.manual_seed(4)
    model=Qwen3_5ForCausalLM(cfg).eval()
    counts=install_all24(model,window=8,anchors=2,chunk_size=4)
    assert counts=={"delta_replaced":18,"full_replaced":6}
    assert sum(isinstance(l.linear_attn,WindowAnchorDelta)
               for l in model.model.layers if l.block_type=="linear_attention")==18
    assert sum(isinstance(l.self_attn,WindowAnchorFull)
               for l in model.model.layers if l.block_type=="full_attention")==6
    with torch.inference_mode():
        x=torch.tensor([[4,5,6,7,8,9,10,11]],dtype=torch.long)
        logits=model(input_ids=x,use_cache=False,logits_to_keep=1).logits
    assert logits.shape==(1,1,128)
    assert torch.isfinite(logits).all()
