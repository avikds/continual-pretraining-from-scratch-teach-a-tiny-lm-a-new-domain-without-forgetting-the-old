"""
Continual Pretraining from Scratch: Teach a Tiny LM a New Domain Without Forgetting the Old scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Continual pretraining of a tiny character-level GPT, from scratch.

Story: the model is pretrained on children's stories (domain A) with a schedule that
decays to a small rate, then continued on Shakespeare (domain B). Naive continuation
at the decayed rate learns B slowly and still forgets some of A. Re-warming the
schedule learns B fast and forgets A, and the old-domain loss climbs and never comes
back. Replaying a quarter of A in every batch keeps A at almost no cost in B. EWC
holds A with a stiff Fisher leash but pays in acquisition; LoRA forgets like full
fine-tuning with the adapters on and recovers A exactly with them off. The straight
line in weight space between the old and new models trades the two losses smoothly.
"""
import torch


def main() -> None:
    torch.manual_seed(0)
    rep = cl_experiment(seed=0, pre_steps=400, cont_steps=150)
    b = rep["base"]
    v = rep["variants"]

    print("1. Two domains, one model")
    print(f"   character vocabulary {rep['vocab']} | TinyGPT with {b['n_params']} parameters | pretraining {rep['pre_steps']} steps on stories, continuation {rep['cont_steps']} steps on Shakespeare")
    print(f"   base after pretraining: stories {b['losses']['A']:.3f} nats/char, Shakespeare {b['losses']['B']:.3f} nats/char (final train loss {b['final_train_loss']:.3f})")
    print("   The gap between the two is what continuation has to close; the stories loss is what it must not lose.")

    print("\n2. Forgetting and acquisition by strategy (nats per character; forgetting = rise on stories, acquisition = drop on Shakespeare)")
    for name in ("naive", "rewarm", "replay", "ewc", "lora"):
        r = v[name]["report"]
        print(f"   {name:7s}: stories {r['A_before']:.3f} -> {r['A_after']:.3f} (forgetting {r['forgetting']:+.3f}) | Shakespeare {r['B_before']:.3f} -> {r['B_after']:.3f} (acquisition {r['acquisition']:+.3f})")
    print("   Re-warming learns the most and forgets the most; naive continuation at the decayed rate learns least.")
    print("   Replaying a quarter of the old domain removes most of the forgetting at almost no cost in acquisition.")

    print("\n3. The old-domain loss during continuation (monitored every 25 steps)")
    for name in ("rewarm", "replay", "ewc"):
        c = v[name]["curve"]
        print(f"   {name:7s}: {' '.join(f'{x:.2f}' for x in c['losses'])} | peak {c['peak']:.3f} at step {c['peak_step']}, rise {c['rise']:+.3f}, final {c['final']:.3f}")
    print("   Without replay the old loss climbs during the high-rate steps and stays up; with replay it bumps early and comes back down.")

    e = v["ewc"]["extra"]["fisher_top"]
    print("\n4. Where the old domain lives (largest mean Fisher entries)")
    for name, val in e:
        print(f"   {name:28s} {val:.3e}")
    print("   The leash is stiffest on the few parameters every prediction passes through: normalisation and the output head.")

    l = v["lora"]
    print(f"\n5. Adapters: {l['extra']['trainable']} trainable parameters, base frozen")
    print(f"   adapters on : stories {l['report']['A_after']:.3f}, Shakespeare {l['report']['B_after']:.3f}")
    print(f"   adapters off: stories {l['extra']['lora_off']['A']:.3f}, Shakespeare {l['extra']['lora_off']['B']:.3f} (the base, exactly)")
    print("   LoRA does not prevent forgetting; it makes it reversible, at the price of knowing which domain an input is from.")

    print("\n6. Merging by interpolation between the old model and the re-warmed one (alpha: stories / Shakespeare)")
    for p in rep["interpolation"]:
        print(f"   alpha {p['alpha']:.2f}: {p['A']:.3f} / {p['B']:.3f}")
    print("   The two losses trade smoothly along the line: an intermediate point is a compromise model nobody trained.")
    print("   Numbers vary a little by seed; the ordering of the strategies does not.")


if __name__ == "__main__":
    main()

