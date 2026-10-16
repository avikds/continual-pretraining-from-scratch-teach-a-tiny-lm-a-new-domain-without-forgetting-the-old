# Continual Pretraining from Scratch: Teach a Tiny LM a New Domain Without Forgetting the Old

A pretrained language model is never finished: it has to absorb new domains, new languages and newer data without losing what it already knows. This project rebuilds that problem at a scale you can run on a CPU. You pretrain a small character-level GPT on children's stories, then continue pretraining it on Shakespeare and measure exactly what happens to the old domain. Naive continuation at the decayed learning rate barely learns the new text; re-warming the schedule learns it fast and forgets the old domain; replaying a quarter of the old data keeps the old loss where it was at almost no cost in acquisition. You then build the other tools of continual learning on the same model: a diagonal Fisher and the Elastic Weight Consolidation penalty, LoRA adapters that leave the base weights untouched so the old domain can be switched back on, and weight-space interpolation between the old and new models that traces a linear path through both losses. The final experiment runs every variant from the same checkpoint and reports forgetting, acquisition and the old-domain loss curve for each, so you see the stability-plasticity trade-off as numbers rather than as a slogan.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** fetch_text
- [x] **2.** CharTokenizer
- [x] **3.** make_batch
- [x] **4.** load_domains
- [x] **5.** TinyGPT
- [x] **6.** lm_loss
- [x] **7.** eval_loss
- [x] **8.** cosine_lr
- [x] **9.** train_phase
- [x] **10.** domain_losses
- [x] **11.** forgetting_report
- [x] **12.** old_loss_curve
- [x] **13.** replay_batch
- [x] **14.** diagonal_fisher
- [x] **15.** ewc_penalty
- [x] **16.** LoRALinear
- [x] **17.** add_lora
- [x] **18.** interpolation_path
- [x] **19.** build_base
- [x] **20.** continue_variant
- [x] **21.** cl_experiment

## Results

```
1. Two domains, one model
   character vocabulary 82 | TinyGPT with 114688 parameters | pretraining 400 steps on stories, continuation 150 steps on Shakespeare
   base after pretraining: stories 1.999 nats/char, Shakespeare 2.906 nats/char (final train loss 1.893)
   The gap between the two is what continuation has to close; the stories loss is what it must not lose.

2. Forgetting and acquisition by strategy (nats per character; forgetting = rise on stories, acquisition = drop on Shakespeare)
   naive  : stories 1.999 -> 2.128 (forgetting +0.130) | Shakespeare 2.906 -> 2.478 (acquisition +0.428)
   rewarm : stories 1.999 -> 2.219 (forgetting +0.220) | Shakespeare 2.906 -> 2.203 (acquisition +0.704)
   replay : stories 1.999 -> 2.037 (forgetting +0.038) | Shakespeare 2.906 -> 2.224 (acquisition +0.682)
   ewc    : stories 1.999 -> 2.074 (forgetting +0.075) | Shakespeare 2.906 -> 2.471 (acquisition +0.435)
   lora   : stories 1.999 -> 2.188 (forgetting +0.190) | Shakespeare 2.906 -> 2.420 (acquisition +0.487)
   Re-warming learns the most and forgets the most; naive continuation at the decayed rate learns least.
   Replaying a quarter of the old domain removes most of the forgetting at almost no cost in acquisition.

3. The old-domain loss during continuation (monitored every 25 steps)
   rewarm : 1.99 2.20 2.23 2.24 2.22 2.22 2.21 | peak 2.238 at step 75, rise +0.252, final 2.207
   replay : 1.99 2.11 2.09 2.08 2.05 2.04 2.03 | peak 2.113 at step 25, rise +0.127, final 2.028
   ewc    : 1.99 2.09 2.09 2.08 2.06 2.07 2.07 | peak 2.091 at step 25, rise +0.105, final 2.066
   Without replay the old loss climbs during the high-rate steps and stays up; with replay it bumps early and comes back down.

4. Where the old domain lives (largest mean Fisher entries)
   blocks.0.ln1.bias            3.555e-05
   head.weight                  2.828e-05
   blocks.0.proj.bias           2.178e-05
   The leash is stiffest on the few parameters every prediction passes through: normalisation and the output head.

5. Adapters: 16384 trainable parameters, base frozen
   adapters on : stories 2.188, Shakespeare 2.420
   adapters off: stories 1.999, Shakespeare 2.906 (the base, exactly)
   LoRA does not prevent forgetting; it makes it reversible, at the price of knowing which domain an input is from.

6. Merging by interpolation between the old model and the re-warmed one (alpha: stories / Shakespeare)
   alpha 0.00: 1.999 / 2.906
   alpha 0.25: 1.991 / 2.646
   alpha 0.50: 2.027 / 2.413
   alpha 0.75: 2.103 / 2.263
   alpha 1.00: 2.219 / 2.203
   The two losses trade smoothly along the line: an intermediate point is a compromise model nobody trained.
   Numbers vary a little by seed; the ordering of the strategies does not.
```
