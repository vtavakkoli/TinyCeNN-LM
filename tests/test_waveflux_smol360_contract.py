"""CPU tests for strict layer-0 attention-slot replacement, no downloads."""
import pytest
import torch
from torch import nn
from scripts.benchmark_waveflux_smol360 import WaveFluxAttentionAdapter, attention_slot
from tinycenn_lm.waveflux_cenn import WaveFluxConfig, WaveFluxCeNN


class NativeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
    def forward(self, hidden_states, **kwargs):
        self.calls += 1
        return torch.full_like(hidden_states,7.0),None


class ToyDecoder(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.self_attn = native
    def forward(self, hidden_states):
        y, _ = self.self_attn(hidden_states=hidden_states)
        return hidden_states+y  # native Llama decoder residual, applied once


class ToyModel(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyDecoder(native)])


def test_replaces_only_layer0_native_never_called():
    native = NativeAttention()
    model = ToyModel(native)
    cfg = WaveFluxConfig(hidden_size=24,groups=6,cell_width=4,velocities=(0,1,-1))
    cell = WaveFluxCeNN(cfg)
    new = WaveFluxAttentionAdapter(cell)
    x = torch.randn(2,13,24)
    with attention_slot(model,new):
        y = model.model.layers[0](x)
        assert native.calls == 0
        torch.testing.assert_close(y-x,cell(x),rtol=0,atol=3e-7)
        assert isinstance(new(x),tuple) and new(x)[1] is None
        with pytest.raises(NotImplementedError):
            new(x,past_key_value=object())
        with pytest.raises(NotImplementedError):
            new(x,output_attentions=True)
    assert model.model.layers[0].self_attn is native
    out = model.model.layers[0](x)
    torch.testing.assert_close(out-x, torch.full_like(x,7.0))
    assert native.calls == 1


def test_restore_even_on_exception():
    native=NativeAttention(); model=ToyModel(native)
    cell=WaveFluxCeNN(WaveFluxConfig(hidden_size=24,groups=6,cell_width=4))
    with pytest.raises(RuntimeError):
        with attention_slot(model,WaveFluxAttentionAdapter(cell)):
            raise RuntimeError("deliberate")
    assert model.model.layers[0].self_attn is native
