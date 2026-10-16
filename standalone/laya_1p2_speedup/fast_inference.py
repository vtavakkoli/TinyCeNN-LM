"""Inference acceleration for the unchanged Laya-1.2 standalone FP16 weights.

Pin transformers==4.57.6. Use optimize_model(original.eval()), then validate on
representative inputs. Compilation/CUDA graphs are opt-in and hardware dependent.
"""
from __future__ import annotations
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F


def inference_context(model):
    return (torch.autocast('cuda', dtype=torch.float16)
            if next(model.parameters()).is_cuda else nullcontext())


def marker_head(head, x, keep, count):
    """Keep every K/V, but compute only marker queries/FFN in the final layer."""
    for i, layer in enumerate(head.layers):
        a = layer.self_attn
        b, s, d = x.shape
        last = i == len(head.layers) - 1
        z = layer.norm1(x)
        residual = x[:, 1:1 + count] if last else x
        if last:
            bias = a.in_proj_bias
            q = F.linear(z[:, 1:1 + count], a.in_proj_weight[:d],
                         None if bias is None else bias[:d])
            k, v = F.linear(z, a.in_proj_weight[d:],
                            None if bias is None else bias[d:]).chunk(2, -1)
        else:
            q, k, v = F.linear(z, a.in_proj_weight, a.in_proj_bias).chunk(3, -1)
        q = q.reshape(b, -1, a.num_heads, d // a.num_heads).transpose(1, 2)
        k = k.reshape(b, s, a.num_heads, d // a.num_heads).transpose(1, 2)
        v = v.reshape(b, s, a.num_heads, d // a.num_heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None],
                                           dropout_p=0., is_causal=False)
        x = residual + a.out_proj(y.transpose(1, 2).reshape(b, -1, d))
        x = x + layer.linear2(layer.activation(layer.linear1(layer.norm2(x))))
    return head.norm(x) if head.norm is not None else x


class FastLaya(nn.Module):
    """Shares weights with the reference. Inference only; does not mutate weights."""
    def __init__(self, model, *, prune_head=True, device_masks=True):
        super().__init__()
        if model.training:
            raise ValueError('Call eval() before optimize_model()')
        if not len(model.head.layers):
            raise ValueError('Empty decision head')
        for layer in model.head.layers:
            a = layer.self_attn
            if (not layer.norm_first or not a.batch_first or not a._qkv_same_embed_dim
                    or a.bias_k is not None or a.bias_v is not None or a.add_zero_attn):
                raise ValueError('Expected the standard Laya pre-norm decision head')
        import transformers
        self.device_masks = (device_masks and transformers.__version__ == '4.57.6'
                             and type(model.encoder).__name__ == 'ModernBertModel'
                             and model.encoder.config._attn_implementation == 'sdpa')
        self.model = model
        self.prune_head = prune_head
        self._compiled_forward = None
        self.eval()

    def train(self, mode=True):
        if mode:
            raise RuntimeError('FastLaya is inference-only')
        return super().train(False)

    def _encode(self, input_ids, attention_mask):
        e = self.model.encoder
        if not self.device_masks:
            return e(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Reproduce 4.57.6's encoder with on-device window construction. The
        # native helper builds a CPU T*T mask and copies it each forward, which
        # obstructs CUDA graph capture. Global key masks broadcast over queries.
        t = input_ids.shape[1]
        positions = torch.arange(t, device=input_ids.device)
        dtype = next(e.parameters()).dtype
        minimum = torch.finfo(dtype).min
        global_mask = torch.zeros_like(attention_mask, dtype=dtype)[:, None, None]
        global_mask = global_mask.masked_fill(~attention_mask[:, None, None].bool(), minimum)
        window = (positions[:, None] - positions[None, :]).abs() <= e.config.local_attention // 2
        local_mask = global_mask.expand(-1, 1, t, -1).masked_fill(~window[None, None], minimum)
        x = e.embeddings(input_ids=input_ids)
        for layer in e.layers:
            x = layer(x, attention_mask=global_mask, sliding_window_mask=local_mask,
                      position_ids=positions[None], output_attentions=False)[0]
        return e.final_norm(x)

    def _forward_impl(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        m = self.model
        h = self._encode(input_ids, attention_mask)
        compact, keep = m._select_head_tokens(h, attention_mask, marker_pos, marker_mask)
        # Identical addition at selected tokens; no full-sequence allocation.
        compact = compact + m.type_emb(qtype)[:, None]
        if self.prune_head:
            selected = marker_head(m.head, compact, keep, marker_pos.shape[1])
        else:
            selected = m.head(compact, src_key_padding_mask=~keep)[:, 1:1 + marker_pos.shape[1]]
        logits = m.scorer(selected).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, -1e4), None

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        fn = self._compiled_forward or self._forward_impl
        return fn(input_ids, attention_mask, marker_pos, marker_mask, qtype)


def optimize_model(model, *, prune_head=True, device_masks=True, compile_mode=None):
    # The source FP16 loader already disables reference_compile. Do not silently
    # change a shared reference model or trainable weights during optimization.
    if getattr(model.encoder.config, 'reference_compile', False) is not False:
        raise ValueError('Load with ModernBERT reference_compile=False')
    fast = FastLaya(model, prune_head=prune_head, device_masks=device_masks)
    if compile_mode:
        fast._compiled_forward = torch.compile(fast._forward_impl, mode=compile_mode, dynamic=False)
    return fast


class PreparedChoices:
    """Cache question tokens; tokenize each state once. Preserve original padded T.

    The runtime argument is the checkpoint's modeling_laya_standalone module.
    Recreate this object when questions or tokenizer change. No encoder-state
    caching: bidirectional attention makes those states input dependent.
    """
    def __init__(self, runtime, tokenizer, questions, *, max_len=512, head_max_len=192):
        if not questions:
            raise ValueError('questions must not be empty')
        self.runtime, self.tokenizer, self.max_len = runtime, tokenizer, max_len
        self.rows = []
        for name, raw in questions.items():
            q = runtime.normalize_question(raw)
            if q['t'] != 'choice':
                raise ValueError('PreparedChoices supports choice questions only')
            ids, markers = runtime.build_sequence(tokenizer, '', q, max_len=1_000_000,
                                                  head_max_len=head_max_len, state_ids=[])
            labels = tuple(q['crit'])
            if len(ids) > max_len or len(markers) != len(labels):
                raise ValueError('max_len truncates options; increase it')
            self.rows.append((name, labels, tuple(ids[:-1]), tuple(markers), runtime.QTYPES[q['t']]))

    def batch(self, state, *, device='cpu'):
        tok = self.tokenizer
        text = self.runtime._serialize_state(state).replace(tok.mask_token, ' ')
        state_ids = tok(text, add_special_tokens=False)['input_ids']
        items = []
        for _, _, prefix, markers, qtype in self.rows:
            room = max(0, self.max_len - len(prefix) - 1)
            items.append(dict(ids=list(prefix) + state_ids[:room] + [tok.sep_token_id],
                              markers=markers, qtype=qtype))
        batch = self.runtime.collate_items(items, int(tok.pad_token_id))
        return {k: v.to(device) for k, v in batch.items()}

    def decode(self, logits, marker_mask):
        probs = logits.float().masked_fill(~marker_mask, -1e9).softmax(-1).cpu()
        answers = {}
        for i, (name, labels, *_rest) in enumerate(self.rows):
            p = probs[i, :len(labels)]
            best = int(p.argmax())
            answers[name] = dict(choice=labels[best], confidence=float(p[best]),
                                 probabilities=dict(zip(labels, map(float, p))))
        return answers

    @torch.inference_mode()
    def predict(self, model, state):
        batch = self.batch(state, device=next(model.parameters()).device)
        with inference_context(model):
            logits, _ = model(**batch)
        return self.decode(logits, batch['marker_mask'])


class CUDAGraphRunner:
    """Capture one exact signature. Includes input copies and owned output clone.

    Not thread-safe; create one runner per worker, and recreate after changing
    weights/dtype/backend. Never repad implicitly: Laya landmarks depend on T.
    """
    @torch.inference_mode()
    def __init__(self, model, example, *, warmup=3):
        self.device = next(model.parameters()).device
        if model.training or self.device.type != 'cuda':
            raise ValueError('CUDA graph requires an eval model on CUDA')
        if getattr(model, '_compiled_forward', None) is not None:
            raise ValueError('Capture eager FastLaya; do not nest compiled CUDA graphs')
        if any(v.device != self.device for v in example.values()):
            raise ValueError('Inputs must be on the model device')
        self.model = model
        self.signature = {k: (v.shape, v.dtype, v.device) for k, v in example.items()}
        with torch.cuda.device(self.device):
            self.inputs = {k: v.clone() for k, v in example.items()}
            stream = torch.cuda.Stream(device=self.device)
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream), inference_context(model):
                for _ in range(warmup):
                    model(**self.inputs)
            torch.cuda.current_stream(self.device).wait_stream(stream)
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph), inference_context(model):
                self.output = model(**self.inputs)[0]

    @torch.inference_mode()
    def __call__(self, **batch):
        if {k: (v.shape, v.dtype, v.device) for k, v in batch.items()} != self.signature:
            raise ValueError('CUDA graph signature mismatch; capture this shape separately')
        with torch.cuda.device(self.device):
            for k, v in batch.items():
                self.inputs[k].copy_(v)
            self.graph.replay()
            return self.output.clone(), None


def make_flash_attention_model(runtime, model):
    """Optional separately constructed FA2 encoder; preserves the reference.

    Requires compatible flash-attn and normally Ampere or newer (not T4).
    Kernel/RoPE numerical differences must pass a workload-level parity gate.
    """
    import copy
    from types import SimpleNamespace
    from transformers import AutoModel
    device = next(model.parameters()).device
    if device.type != 'cuda' or torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError('FlashAttention-2 requires a supported GPU (Ampere or newer)')
    cfg = copy.deepcopy(model.encoder.config)
    cfg.reference_compile = False
    # Backend changes the rotary module at construction, so do not just toggle
    # _attn_implementation on an already constructed encoder.
    with torch.device(device):
        encoder = AutoModel.from_config(cfg, attn_implementation='flash_attention_2',
                                        torch_dtype=torch.float16).half()
    encoder.load_state_dict(model.encoder.state_dict(), strict=True)
    parts = SimpleNamespace(encoder=encoder, head=model.head,
                            type_emb=model.type_emb, scorer=model.scorer)
    return runtime.LayaStandalone(budget=model.budget, pretrained_components=parts).eval()
