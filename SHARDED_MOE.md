# Compact Sharded MoE-CeNN

This experiment replaces the earlier 8-full-expert MoE with a parameter-neutral routed FFN.

## Design

The trained plain CeNN FFN has hidden size 192 and expansion 4, so its SwiGLU inner width is 768. Instead of cloning that full FFN eight times, the inner dimension is split into eight disjoint 96-channel shards.

```text
CeNN local state
      |
      v
  complete FFN
  768 inner channels
      |
  split exactly
      |
+-----+-----+-----+-----+-----+-----+-----+-----+
| 96  | 96  | 96  | 96  | 96  | 96  | 96  | 96  |
+-----+-----+-----+-----+-----+-----+-----+-----+
   \_________________________________________/
                 complete sum
                       +
             Top-2 routed correction
```

All shard parameters together equal one dense FFN. The only additional trainable parameters are the 192→8 router (1,536 weights) and one route-mix scalar.

For the standard TinyCeNN configuration:

- plain CeNN trainable parameters: 480,192
- sharded routed CeNN trainable parameters: 481,729
- overhead: ~0.32%
- shards: 8
- shard inner width: 96
- active routed shards: Top-2
- recurrent steps: 7
- receptive field: 255

## Warm start

`TinyCeNN-LM-Distilled-v2` is used as the source checkpoint. The trained dense FFN is sliced exactly across the eight shards. Since `route_mix` starts at zero, summing all shards exactly reconstructs the original dense FFN before routed specialization begins.

The trainer aborts if held-out CE differs from the plain checkpoint by more than the configured warm-start tolerance.

## One-hour Colab budget

The default Colab run uses:

- 20M additional tokens
- learning rate 2e-4
- 65,536 held-out tokens
- evaluation every 500 updates
- 50-minute training-loop wall-clock cap

The 50-minute cap leaves margin for final evaluation and Hugging Face upload/reload inside an approximately one-hour Colab workflow.

## Colab

`notebooks/TinyCeNN_SharedFFN_Top2_Colab.ipynb`

The notebook publishes the best checkpoint as `<HF-user>/TinyCeNN-LM-Sharded-MoE-Top2` and re-downloads it to reproduce the exact held-out benchmark fingerprint and CE.
