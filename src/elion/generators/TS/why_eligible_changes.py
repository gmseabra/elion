"""
Why the "Slot N eligible" number (e.g. your 34875) changes.

This uses your REAL disallow_tracker.DisallowTracker — no reimplementation —
to reproduce the exact quantity the sidebar shows. In thompson_sampling.py the
search loop computes, for the slot it is about to fill:

    disallow_mask = self._disallow_tracker.get_disallowed_selection_mask(selected_reagents)
    competitors   = len(reagent_list) - len(disallow_mask)      # line ~467

`competitors` is what the UI labels "Slot N eligible" (tsTsPoolN_jX). We show
two independent reasons it is not a fixed number.
"""
import random
import numpy as np
from disallow_tracker import DisallowTracker

# Small 2-site library so the numbers are readable.
# (Your real run: a slot can hold tens of thousands of reagents -> ~34,875.)
SLOT_SIZES = [12, 8]                       # slot 0: 12 reagents, slot 1: 8
random.seed(0)                            # seed ONLY the demo, for reproducibility
np.random.seed(0)

tr = DisallowTracker(SLOT_SIZES)
print(f"Library: slot0={SLOT_SIZES[0]}, slot1={SLOT_SIZES[1]} reagents, "
      f"{int(np.prod(SLOT_SIZES))} possible products\n")


def eligible(tracker, fixed):
    """eligible-pool size for the single open (None) slot, given the rest of
    the selection -- exactly what search() does before picking a winner."""
    slot = fixed.index(None)
    return SLOT_SIZES[slot] - len(tracker.get_disallowed_selection_mask(fixed))


# ── Part A: run the actual selection pattern and watch the numbers ──────────
print("PART A  -- 'Slot N eligible' as the search loop actually produces it")
print("          (each row is one iteration; this is the number on screen)\n")
print(f"{'iter':>4} | {'fill order':>11} | {'slot0 elig':>10} | {'slot1 elig':>10}")
print("-" * 48)
for it in range(14):
    selected = [DisallowTracker.Empty] * 2
    order = random.sample(range(2), 2)            # mirrors random.sample() in search()
    snap = {}
    for cyc in order:
        selected[cyc] = DisallowTracker.To_Fill
        mask = tr.get_disallowed_selection_mask(selected)
        snap[cyc] = SLOT_SIZES[cyc] - len(mask)   # <-- the displayed value
        choices = [i for i in range(SLOT_SIZES[cyc]) if i not in mask]
        selected[cyc] = random.choice(choices)    # stand-in for the random TS draw
    tr.update(selected)
    print(f"{it:>4} | {str(order):>11} | {snap[0]:>10} | {snap[1]:>10}")

# ── Part B: WHY it moves -- the count is conditional on the partner ─────────
print("\nPART B  -- the same slot's eligible count depends on which partner")
print("          is already fixed this iteration (partners are random draws)\n")
for partner in [DisallowTracker.Empty, 0, 1, 2, 3, 4]:
    e = eligible(tr, [None, partner])
    where = "slot1 still open" if partner == DisallowTracker.Empty else f"slot1 = reagent {partner}"
    print(f"  slot0 eligible = {e:>3}   when {where}")

# ── Part C: permanent shrink -- retiring reagents (failed reactions) ────────
print("\nPART C  -- retiring reagents permanently lowers the pool (monotonic)")
before = eligible(tr, [DisallowTracker.Empty, None])   # slot1 unconstrained
tr.retire_one_synthon(1, 5)                            # e.g. slot1 reagent 5 always fails
tr.retire_one_synthon(1, 6)
after = eligible(tr, [DisallowTracker.Empty, None])
print(f"  slot1 eligible (unconstrained): {before}  ->  {after}  "
      f"after retiring 2 reagents")