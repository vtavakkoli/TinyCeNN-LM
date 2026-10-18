# WaveFlux-CeNN: first self-attention replacement on SmolLM2-360M-Instruct

## Objective

**Test one genuinely attention-free cellular module** in the first `LlamaDecoderLayer.self_attn` of [`HuggingFaceTB/SmolLM2-360M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct).
The module runs only WaveFlux's locally transported persistent cellular states, scan-compatible gated updates, conservative local readout and projections. **It has no attention, Q/K/V, DeltaNet or LSTM implementation or bypass.** The pretrained 360M network outside that single attention slot remains in place as an experimental harness: **the complete language model is still hybrid until every attention layer is replaced and validated.** This experiment does not claim full standalone LM replacement.

Notebook: [`notebooks/WaveFlux_SmolLM2_360M_Layer0_Standalone_Colab.ipynb`](../notebooks/WaveFlux_SmolLM2_360M_Layer0_Standalone_Colab.ipynb)  
Runner: [`scripts/benchmark_waveflux_smol360.py`](../scripts/benchmark_waveflux_smol360.py)  
Cell: [`src/tinycenn_lm/waveflux_cenn.py`](../src/tinycenn_lm/waveflux_cenn.py)

## Correctness

- Target **layer index 0**, specifically `model.model.layers[0].self_attn`; preserve input-layernorm, outer residual, post-attention norm, FFN, embeddings, later layers and LM head.
- Replace the attention *submodule* output, not the full decoder residual output. Do **not** add a second residual.
- During inference the replacement never computes the old Q/K/V, attention, SDPA or a teacher blend. The original native layer is restored only when evaluating the true original baseline.
- Use `use_cache=False`. Native Llama cache APIs, beam search and `.generate()` are unsupported with this adapter. Neither the WaveFlux state nor its streaming API automatically implements Hugging Face `Cache`.
- Fixed-size **unpadded** causal token streams only. Masked padding and variable-length packing need explicit handling and parity tests. Position embeddings passed by Llama are deliberately not used by WaveFlux, which uses its own cellular position dynamics.
- Layer code is a reference PyTorch scan, not a fused CUDA/Triton accelerator. Real GPU latency must be measured; a faster model is **not** implied.

## Controlled training/evaluation

1. Freeze the original SmolLM2-360M-Instruct from an exact Hugging Face revision and collect its **real layer-0 attention input and output** on a WikiText-2 **train** partition. One training sample is an unpadded token block, capturing only the attention-submodule I/O.
2. Train *only* WaveFlux parameters on relative MSE plus local cosine, with no native-attention mixing or original Q/K/V usage in the candidate.
3. Collect distinct WikiText-2 **validation** blocks; choose the **minimum full-model validation NLL** checkpoint, not minimum local training loss.
4. Load the untouched WikiText-2 **test** partition only after checkpoint selection. Compare all available next-token positions, not sampled positions, with unmodified native 360M inference.
5. Measure matched batch/context full-model prefill times using CUDA events and alternating baseline/student runs. Unfused PyTorch may be slower than optimized SDPA; report such negative results.
6. Save a labelled experimental adapter checkpoint and full metrics, but accept the candidate only when **all** quality and CUDA-speed gates pass. Default thresholds: dev/test ΔNLL ≤0.02 nats/token, teacher argmax agreement ≥0.98, dev local cosine ≥0.95 and paired full-model prefill speedup >1.0x. A test-set failure must not be used for iterative hyperparameter selection.

These criteria are demanding, especially with **only one newly trained mixer**. The first result is a *feasibility experiment*, not an assumption that replacement will work or speed up.

## Run

Choose a CUDA-enabled Colab runtime. Run the notebook from the beginning, or:

```bash
pip install -e .
python scripts/benchmark_waveflux_smol360.py --profile quick \
  --output-dir results/waveflux_smol360_l0
```

Use `--profile smoke` for integration checks. `--profile research` increases context, training and held-out sample budgets. `--skip-gpu-required` supports **CPU-only diagnostic** runs but never earns a speed/acceptance label.

Expected output: `best_waveflux_layer0.pt` (research checkpoint, **not** a validated or deployable 360M model), `dev_history.json`, `dev_history.csv`, and `final_report.json`. The Colab packages these files into a ZIP and offers a download. If `accepted` is false, leave native attention in place.

## Important engineering caveats

- CPU numerical tests verify scan vs recurrent equivalence, causality, transport, finite gradients, chunked streaming and the layer-0 adapter contract. GPU performance tests only run when using a real CUDA runtime.
- Whole-model CUDA timings include all downstream frozen model work, but separate baseline/student instances are **not** used for GPU memory profiling in this version: the native layer and WaveFlux adapter may both be resident. Do not report peak allocator bytes as an independently deployable model's peak memory.
- WikiText-2 splits are separate, but they are **not evidence of non-overlap with pretrained-model pretraining**. A new task-disjoint external evaluation is required for publication.
- The `best_waveflux_layer0.pt` contains only the candidate's weights and metadata, *not* the native model or tokenizer. For future deployment, add checkpoint-loading, cache compatibility and strict parity first.
- **Do not claim scientific novelty** without checking wave-memory recurrence, affine-scan SSMs, temporal cellular automata, xLSTM and other existing approaches. The specific cell-transport combination is a hypothesis.
