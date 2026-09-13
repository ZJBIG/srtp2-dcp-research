# Counterfactual missing-modality game

This experimental branch replaces DCP's final zero feature for a missing
modality with a learned support proxy and trains a bounded challenge proxy
through gradient reversal.

## What is implemented

- `image -> text` and `text -> image` directional proxy generators.
- Direction identity and missing-state identity embeddings.
- Feature-wise gates controlling the source-to-target message.
- A support proxy used by the ordinary task prediction.
- A bounded challenge residual used by the robust prediction.
- Gradient reversal on the challenge residual: the classifier minimizes the
  challenge loss while the residual generator maximizes it.
- Complete-pair cosine alignment and support/challenge plausibility losses.
- MM-IMDb, Hateful Memes, and Food101 classification objectives.
- The original DCP path remains available when the named config is omitted.

This first version operates on final CLIP features. It is intentionally a
diagnosable baseline before moving the game into every Transformer layer.

## Training

From `code/Deep_Correlated_Prompting-main`:

```powershell
python run.py with task_finetune_mmimdb counterfactual_game `
  data_root=D:\path\to\mmimdb\arrow `
  missing_table_root=D:\path\to\missing_tables `
  clip_cache_root=D:\path\to\clip_cache `
  num_gpus=1 per_gpu_batchsize=4
```

Omit `counterfactual_game` to run the original DCP behavior.

## Main hyperparameters

| Config key | Default | Meaning |
| --- | ---: | --- |
| `counterfactual_hidden_dim` | 512 | Proxy generator capacity |
| `counterfactual_identity_dim` | 32 | Direction/missing identity dimension |
| `counterfactual_challenge_radius` | 0.5 | Maximum feature-space challenge radius |
| `counterfactual_adversary_strength` | 0.25 | Reversed gradient scale |
| `counterfactual_robust_weight` | 0.25 | Challenge classification loss weight |
| `counterfactual_alignment_weight` | 0.1 | Complete-pair proxy alignment weight |
| `counterfactual_plausibility_weight` | 0.1 | Challenge-to-support constraint weight |
| `counterfactual_gain_weight` | 0.1 | Complete-vs-unimodal gain loss weight |
| `counterfactual_gain_margin` | 0.05 | Required loss advantage of real bimodal input |
| `counterfactual_lr_mult` | 0.1 | Proxy learning rate relative to base LR |
| `counterfactual_adversary_warmup_steps` | 500 | Linear adversary warm-up |

A conservative initial sweep is:

- radius: `0.25, 0.5, 1.0, 2.0`
- adversary strength: `0.1, 0.25, 0.5, 1.0`
- robust weight: `0.1, 0.25, 0.5, 1.0`
- alignment weight: `0.05, 0.1, 0.25, 0.5`

Do not increase radius and adversary strength simultaneously at the beginning.
If the point F1 collapses, first lower adversary strength; if challenge and
point logits are indistinguishable, increase radius or robust weight.

## Returned values

With the game enabled, `infer` returns:

- `cls_feats`: available feature plus support proxy;
- `challenge_cls_feats`: available feature plus adversarial proxy;
- `counterfactual_mask`: rows with a missing modality;
- `proxy_alignment_loss` and `proxy_plausibility_loss`;
- `proxy_gate_mean` for collapse monitoring;
- both directional support proxies.

Each classification objective also returns `<task>_challenge_logits`. Existing
task metrics continue to use the support prediction so results remain directly
comparable with DCP.

## Current scope and next architectural step

The implementation is an amortized local adversarial searcher: a support point
plus one learned bounded challenge direction. It neither searches every
plausible completion nor estimates certified class-wise probability bounds.
`<task>_challenge_logits` must therefore not be reported as strict lower/upper
bounds. The current cosine constraint is a local latent-space constraint, not a
semantic naturalness guarantee or a learned conditional energy model.

The next large change, after this version establishes a gain, is to insert
support/challenge proxy tokens at selected CLIP Transformer depths and share the
same final losses. That experiment should be compared against this feature-level
version to show whether deep counterfactual propagation is actually necessary.
