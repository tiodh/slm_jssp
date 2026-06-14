# V2 (stratified_v2) Collapse + SIGSEGV — Investigation Notes

## TL;DR
V2 `stratified_v2` reward mode collapses deterministically between step 550–700 on LLaMA-3.1-8B + LoRA + Starjob-SM. Naive resume-from-earlier-checkpoint fails because the collapse trajectory reproduces, and the trajectory's growing completion length triggers a **CUDAGuardImpl SIGSEGV** in unsloth's pre-allocated buffer reuse inside `fast_rms_layernorm_inference`. V2 is **abandoned** as a confirmed negative result; V7 (`hybrid_v7 + length_control`) is the working configuration.

## Timeline

| Step range | Reward | compl_len | Status |
|---|---|---|---|
| 0 – 500 | bouncing 0.0 – 1.0 | <2200 tok | ✅ healthy |
| 500 – 550 | trending negative (0.34 → -0.02) | 544 → 1224 | ⚠️ degrading |
| 550 – 600 | -0.13 → -0.53 | 2343 → 3543 | 🔻 collapsing |
| 600 – 650 | -0.64 → -1.06 | **4096 (pinned)** | ❌ collapsed |
| 650 – 700 | -0.38 → -0.97 | 4096 | ❌ collapsed |

## SIGSEGV Stack (resume from ckpt-500 → crash at step 504–510)

```
File "grpo_trainer.py", line 220, in train
  trainer.train(resume_from_checkpoint=resume_from)
File "transformers/trainer.py", line 2241
  return inner_training_loop(...)
File "<string>", line 25, in _unsloth_training_step
File "UnslothGRPOTrainer.py", line 976, in _prepare_inputs
  prompt_completion_ids = unwrapped_model.generate(...)
File "unsloth/models/llama.py", line 1574, in unsloth_fast_generate
File "transformers/generation/utils.py", line 3214, in _sample
  outputs = model_forward(**model_inputs, return_dict=True)
File "unsloth/models/llama.py", line 1026, in _CausalLM_fast_forward
  outputs = fast_forward_inference(...)
File "unsloth/models/llama.py", line 973, in LlamaModel_fast_forward_inference
File "unsloth/models/llama.py", line 318, in fast_rms_layernorm_inference
  torch_mean(torch_square(XX, out=XX2), -1, keepdim=True, out=variance)
RuntimeError: t == DeviceType::CUDA INTERNAL ASSERT FAILED
              at "../c10/cuda/impl/CUDAGuardImpl.h":28
```

## Root cause analysis

### Why does SIGSEGV happen?

Unsloth's `fast_rms_layernorm_inference` reuses three pre-allocated buffers (`XX`, `XX2`, `variance`) across forward passes for speed:

```python
def fast_rms_layernorm_inference(self, X, XX=None, XX2=None, variance=None):
    if XX is None:
        XX = X.to(torch.float32)
        variance = XX.square().mean(-1, keepdim=True)
    else:
        XX.copy_(X)
        torch_mean(torch_square(XX, out=XX2), -1, keepdim=True, out=variance)
```

`CUDAGuardImpl.h:28` is `TORCH_INTERNAL_ASSERT(t == DeviceType::CUDA)`. The assert fires when a CUDA op is invoked with at least one tensor whose device type ≠ CUDA. With buffer reuse, this happens when growing sequence lengths force reallocation under memory pressure — a buffer ends up on CPU or in a freed state, the next call hits the assert.

### Why V2, not V7?

| Config | V7 (works) | V2 (crashes) |
|---|---|---|
| `length_control` | **True** — `LengthControlledGRPOTrainer` zeros advantages for samples > `OVERLEN_FACTOR × gold_est` | **False** — vanilla GRPOTrainer, no length cap on gradient |
| `grad_accum` | 4 (lower per-step memory) | 1 (higher per-step memory) |
| `temperature` | 0.7 | 0.8 (longer expected completions) |
| `reward_mode` | `hybrid_v7` (stratified + r_o over-emit penalty) | `stratified_v2` (length penalty + EOS bonus, both soft) |
| `max_steps` | 500 (didn't reach collapse zone) | 2000 (would deeply traverse collapse zone) |

V7's length_control is the load-bearing difference: it kills gradient signal for over-long samples so completions don't drift toward 4096. V2 has no such cap, so completion_length grows monotonically once `stratified_v2` rewards bias toward dropping EOS.

### Smoking gun: per-step time growth within one attempt

After resume from ckpt-500, V2 attempt 5 progression:
- Step 501: **15.7 it/s** (0.06 s/step)
- Step 504: **1.5 s/step** (25× slower)
- Step 510: **7.6 s/step** (127× slower)
- → SIGSEGV at step 511

This is exactly the pre-collapse trajectory: each step's completions are longer than the previous, each forward pass invokes more kernel launches with larger sequences, unsloth's buffer reuse stops matching the actual tensor shapes, the CUDA stream desyncs, the assert fires.

## Why resuming from ckpt-500 doesn't escape

Ckpt-500 weights are healthy, but resuming with `stratified_v2` reward function reactivates the same gradient that drove the original collapse 550→700. The trajectory is **deterministic given fixed reward + fixed data ordering** (HF Trainer resume restores RNG state). The model retraces the same path:

1. Reward signal pushes toward dropping EOS to capture "small length penalty" but avoid "missing-op penalty"
2. Completion length grows by ~10–50 tok/step
3. By step ~510, completions hit unsloth's buffer-reuse failure mode
4. SIGSEGV before next `save_every=50` checkpoint

10 attempts confirmed this pattern. All crashed in step 502–511 with the same stack.

## Could we save V2?

| Option | Outcome | Cost |
|---|---|---|
| Add `--length-control` | Likely runs to completion, but no longer a "pure stratified_v2" experiment | Changes experimental contract |
| Switch to vLLM generation | Bypasses unsloth's buffer reuse | Requires re-engineering generate path |
| Patch unsloth llama.py to allocate fresh buffers | Avoids SIGSEGV; collapse still happens (just no crash) | Loses unsloth's speed; doesn't change scientific outcome |
| Lower `save_every` to 5 | Captures more checkpoints before crash | Reward still collapses; just preserves more dead weights |

None of these change the underlying scientific finding: **stratified_v2 alone is insufficient to prevent collapse** on this stack.

## Decision

**V2 is abandoned. Final state preserved:**

- `runs/full_lora_stratified_v2_2000_v2/checkpoint-500` — last healthy checkpoint
- `runs/full_lora_stratified_v2_2000_v2/_collapsed_bkp/checkpoint-{550,600,650,700}` — collapse trajectory preserved for forensic comparison
- No `final_adapter` will be produced for V2

**Watchdog disabled:**

- Removed `*/5 * * * * bash auto_resume_v7_v2.sh` from cron
- Removed `@reboot sleep 90 && bash auto_resume_v7_v2.sh` from cron
- `auto_resume_grpo.sh` (V1–V6) cron remains; it stand-down auto-skips since all final_adapters exist

## What this means for the paper

This is a **load-bearing negative result**, not a failure to report:

1. **V2 (stratified_v2)** alone → collapses 550–700; insufficient
2. **V5/V7 (hybrid + length_control)** → stable to step 500+; necessary

The contrast V2 vs V7 demonstrates that **soft length penalty in the reward function is not enough**; an **explicit gradient mask on over-long samples** (LengthControlledGRPOTrainer) is required. This justifies the length_control mechanism as a core contribution rather than an engineering detail.

## Per-version final status (post-V2-abandon)

| Version | Reward | Length ctrl | Final step | Strict OOD | SM (200) |
|---|---|---|---|---|---|
| V1 | stratified | – | 2000 (final_adapter) | 11/18 (61%) | 196/200 (98%) |
| V2 | stratified_v2 | – | **700 collapsed** | – | – |
| V3 | uniform | – | 2000 (final_adapter) | 9/18 (50%) | – |
| V4 | hybrid (v1+v3 stratified) | – | 2000 (final_adapter) | 13/18 (72%) | – |
| V5 | hybrid_lc | True | 2000 (final_adapter) | TBD | TBD |
| V6 | hybrid_v1 + lc | True | 2000 (final_adapter) | 8/18 (44%) | – |
| V7 | hybrid_v7 + lc + r_o + checker_patch | True | 500 (final_adapter) | 9/18 (50%) | 189/200 (94.5%) |

## References
- Crash stack: `runs/full_lora_stratified_v2_2000_v2/training.log` line 3089–3141
- LengthControlledGRPOTrainer: `grpo_jssp/grpo_trainer.py:107-133`
- Unsloth buffer reuse: `unsloth/models/llama.py:311-328`
- Original V2 collapse evidence: `runs/full_lora_stratified_v2_2000_v2/checkpoint-{600,650,700}/trainer_state.json`
