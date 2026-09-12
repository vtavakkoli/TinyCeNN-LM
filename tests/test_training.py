import json

import pytest
import torch
from torch import nn

from tinycenn_lm import (
    CeNNConfig, freeze_student_interfaces, load_cenn_student_weights,
    replace_transformer_with_cenn, save_cenn_student,
)
from tinycenn_lm.distill_utils import buffered_shuffle, token_blocks
from tinycenn_lm.moe import (
    MoECeNNConfig, freeze_moe_student_interfaces, load_moe_cenn_student_weights,
    replace_transformer_with_moe_cenn, save_moe_cenn_student, warmstart_moe_from_plain_cenn,
)
from tinycenn_lm.training import (
    annealed_weight, checkpoint_training_tokens, lr_multiplier,
    optimizer_groups, plateau_summary, stream_resume_offset,
)
from test_student import DummyModel


class InterfaceModel(DummyModel):
    def __init__(self):
        super().__init__()
        self.model.norm = nn.LayerNorm(8)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head


def student():
    model = InterfaceModel()
    cfg = CeNNConfig(hidden_size=8, steps=2, expansion=2)
    replace_transformer_with_cenn(model, cfg)
    return model, cfg


def test_fp32_master_weights_retain_small_adam_updates():
    # This reproduces the quantization failure, without needing a GPU. Gradients
    # exist in both cases, but bf16 storage cannot represent an update this small.
    def train(dtype):
        p = nn.Parameter(torch.ones(8, dtype=dtype))
        opt = torch.optim.AdamW([p], lr=3e-5, weight_decay=0)
        for _ in range(20):
            opt.zero_grad()
            p.float().square().sum().backward()
            opt.step()
        return p.detach().float()
    assert torch.equal(train(torch.bfloat16), torch.ones(8))
    assert torch.all(train(torch.float32) < 0.9995)


def test_parameter_groups_preserve_interface_lr_and_exclude_norm_decay():
    model, _ = student()
    freeze_student_interfaces(model, "all")
    groups = optimizer_groups(model, 3e-4, 0.05, 0.01)
    entries = {id(p): group for group in groups for p in group["params"]}
    assert len(entries) == len(list(model.parameters()))
    assert entries[id(model.embed.weight)]["lr"] == pytest.approx(1.5e-5)
    assert entries[id(model.model.norm.weight)]["weight_decay"] == 0
    core = model.model.layers[0].cenn.cell
    assert entries[id(core.in_proj.weight)]["lr"] == 3e-4
    assert entries[id(core.gate_proj.bias)]["weight_decay"] == 0
    model.bfloat16()
    with pytest.raises(ValueError, match="must be float32"):
        optimizer_groups(model, 3e-4, 0.05, 0.01)


@pytest.mark.parametrize("scope", ["none", "norm", "all"])
def test_interface_checkpoint_survives_reload_and_refreeze(tmp_path, scope):
    model, cfg = student()
    freeze_student_interfaces(model, scope)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(0.123)
    save_cenn_student(model, tmp_path / "first", config=cfg)
    first = torch.load(tmp_path / "first/cenn_student.pt", weights_only=True)
    assert ("embed.weight" in first) == (scope == "all")
    assert ("model.norm.weight" in first) == (scope != "none")
    restored, _ = student()
    freeze_student_interfaces(restored, "none")
    load_cenn_student_weights(restored, tmp_path / "first")
    save_cenn_student(restored, tmp_path / "second", config=cfg)
    second = torch.load(tmp_path / "second/cenn_student.pt", weights_only=True)
    assert first.keys() == second.keys()
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_corrupt_interface_checkpoint_is_rejected_before_loading(tmp_path):
    model, cfg = student()
    freeze_student_interfaces(model, "all")
    save_cenn_student(model, tmp_path, config=cfg)
    state = torch.load(tmp_path / "cenn_student.pt", weights_only=True)
    del state["lm_head.weight"]
    torch.save(state, tmp_path / "cenn_student.pt")
    with pytest.raises(RuntimeError, match="missing CeNN student keys"):
        load_cenn_student_weights(model, tmp_path)


def test_legacy_core_checkpoint_loads_into_unfrozen_student(tmp_path):
    model, cfg = student()
    freeze_student_interfaces(model)
    save_cenn_student(model, tmp_path, config=cfg)
    path = tmp_path / "student_config.json"
    metadata = json.loads(path.read_text())
    metadata.pop("state_keys")
    metadata["format_version"] = 1
    path.write_text(json.dumps(metadata))
    freeze_student_interfaces(model, "all")
    load_cenn_student_weights(model, tmp_path)


def test_moe_warmstart_and_reload_keep_adapted_interfaces(tmp_path):
    dense, cfg = student()
    freeze_student_interfaces(dense, "all")
    with torch.no_grad():
        dense.embed.weight.add_(1)
        dense.lm_head.weight.sub_(2)
        dense.model.norm.weight.mul_(3)
    save_cenn_student(dense, tmp_path / "dense", config=cfg)
    moe = InterfaceModel()
    moe_cfg = MoECeNNConfig(hidden_size=8, steps=2, expansion=2)
    replace_transformer_with_moe_cenn(moe, moe_cfg)
    warmstart_moe_from_plain_cenn(moe, tmp_path / "dense")
    freeze_moe_student_interfaces(moe)
    save_moe_cenn_student(moe, tmp_path / "moe", config=moe_cfg)
    restored = InterfaceModel()
    replace_transformer_with_moe_cenn(restored, moe_cfg)
    load_moe_cenn_student_weights(restored, tmp_path / "moe")
    torch.testing.assert_close(restored.embed.weight, dense.embed.weight)
    torch.testing.assert_close(restored.lm_head.weight, dense.lm_head.weight)
    torch.testing.assert_close(restored.model.norm.weight, dense.model.norm.weight)


class IntegerTokenizer:
    eos_token_id = 999

    def __call__(self, text, **kwargs):
        return {"input_ids": [int(x) for x in text.split()]}


def test_resumed_stream_matches_uninterrupted_stream_across_block_sizes():
    rows = [{"text": f"{i} {i + 1} {i + 2}"} for i in range(100)]
    def blocks(size, skip=0):
        shuffled = buffered_shuffle(rows, buffer_size=7, seed=42)
        return list(token_blocks(shuffled, IntegerTokenizer(), "text", size, skip_tokens=skip))
    uninterrupted = torch.cat(blocks(4))
    resumed = torch.cat(blocks(7, 53))  # inside a document and a prior block
    torch.testing.assert_close(resumed, uninterrupted[53:53 + len(resumed)])
    assert not blocks(4, 1000)


def test_best_checkpoint_count_and_cursor_ignore_later_run_report():
    signature = {"dataset": "test", "seed": 42}
    metadata = {"training": {"cumulative_tokens": 123, "tokens_this_run": 23,
                              "data_stream": {"signature": signature, "next_token_offset": 400}}}
    report = {"cumulative_training_tokens": 1000, "seen_tokens_this_run": 900}
    assert checkpoint_training_tokens(metadata, report) == 123
    assert stream_resume_offset(metadata, report, signature) == (400, "exact_token_offset")
    with pytest.raises(ValueError, match="resume data stream differs"):
        stream_resume_offset(metadata, report, {"dataset": "changed"})
    assert stream_resume_offset(metadata, report, {}, explicit_offset=0) == (0, "explicit")
    del metadata["training"]["data_stream"]
    assert stream_resume_offset(metadata, report, signature) == (23, "legacy_estimate")


def test_wsd_stays_at_peak_until_decay_and_reaches_floor():
    values = [lr_multiplier(i, 100, 10, schedule="wsd") for i in range(1, 101)]
    assert values[0] == pytest.approx(0.1)
    assert values[9:80] == [1.0] * 71
    assert all(a >= b for a, b in zip(values[79:], values[80:]))
    assert values[-1] == pytest.approx(0.1)
    assert annealed_weight(1, 0.25, 1) == 0.25
    assert annealed_weight(0.25, 0, 1) == 0
    assert annealed_weight(0.25, None, 1) == 0.25


def test_plateau_detects_recent_stagnation_despite_early_progress():
    stalled = [{"student_ce": x} for x in (6, 5, 4.8, 4.8001, 4.7999, 4.8, 4.8)]
    assert plateau_summary(stalled, 4, 0.002)["detected"]
    improving = [{"student_ce": x} for x in (6, 5, 4.8, 4.7, 4.6)]
    assert not plateau_summary(improving, 4, 0.002)["detected"]
