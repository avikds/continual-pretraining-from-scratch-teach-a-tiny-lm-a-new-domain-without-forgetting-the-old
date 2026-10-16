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

---

Built on Deep-ML.
