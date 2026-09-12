import torch
from torch import nn

from tinycenn_lm.story_v2 import CausalStoryMemory, LowRankLMHeadAdapter, StoryV2Config


def test_story_v2_config_defaults_are_small():
    cfg = StoryV2Config()
    assert cfg.memory_rank == 32
    assert cfg.head_rank == 4


def test_causal_story_memory_is_function_preserving_at_init():
    torch.manual_seed(0)
    memory = CausalStoryMemory(hidden_size=12, rank=3)
    x = torch.randn(2, 7, 12)
    y = memory(x)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-7)


def test_causal_story_memory_has_no_future_leakage():
    torch.manual_seed(1)
    memory = CausalStoryMemory(hidden_size=8, rank=2)
    with torch.no_grad():
        memory.up.weight.normal_(0.0, 0.1)
    x1 = torch.randn(1, 6, 8)
    x2 = x1.clone()
    x2[:, 4:, :] += 100.0
    y1 = memory(x1)
    y2 = memory(x2)
    assert torch.allclose(y1[:, :4], y2[:, :4], atol=1e-5)


def test_low_rank_head_is_exact_noop_at_init():
    torch.manual_seed(2)
    base = nn.Linear(10, 17, bias=False)
    adapter = LowRankLMHeadAdapter(base, hidden_size=10, vocab_size=17, rank=3)
    x = torch.randn(2, 5, 10)
    assert torch.allclose(adapter(x), base(x), atol=1e-7)


def test_added_parameter_budget_matches_design():
    h, vocab, memory_rank, head_rank = 192, 32000, 32, 4
    memory = CausalStoryMemory(h, memory_rank)
    base = nn.Linear(h, vocab, bias=False)
    head = LowRankLMHeadAdapter(base, h, vocab, head_rank)
    memory_params = sum(p.numel() for p in memory.parameters())
    head_params = sum(p.numel() for p in head.down.parameters()) + sum(
        p.numel() for p in head.up.parameters()
    )
    assert memory_params == 2 * h * memory_rank
    assert head_params == h * head_rank + vocab * head_rank
    assert 481_729 + memory_params + head_params < 650_000
