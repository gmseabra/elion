"""
Why "Slot N eligible" goes UP as well as down — with ZERO reagents retired.

The number on screen is the per-cycle `competitors` value:
    competitors = slot_size - len(get_disallowed_selection_mask(current_selection))
It is CONDITIONAL on the partner already fixed this iteration. Since Thompson
sampling re-draws the partner every iteration (and the fill order is random),
the value you catch on any given refresh bounces — even though the permanent
mask only ever grows.

This sim has NO failed reactions, so NOTHING is ever permanently retired.
The 'CEILING' column (true reagents remaining) is therefore flat. The 'LIVE'
column (what the sidebar shows) still moves both directions.
"""
import random
import numpy as np
from disallow_tracker import DisallowTracker

random.seed(7)
np.random.seed(7)

S0, S1 = 200, 80                       # slot sizes (small, readable)
tr = DisallowTracker([S0, S1])

# Slot-1 reagents get posterior means so Thompson sampling favours a few of them
# (high-mu partners win often -> they accumulate many combos -> bigger dips).
mu1  = np.random.normal(5.0, 1.0, S1)
std1 = np.full(S1, 0.6)


def ceiling_slot0():
    """True reagents remaining for slot 0 = slot_size - permanently masked.
    (partner left OPEN, so no per-iteration conditioning)"""
    return S0 - len(tr.get_disallowed_selection_mask([None, DisallowTracker.Empty]))


print(f"{'iter':>4} | {'slot0 LIVE (on screen)':>22} | {'slot0 CEILING (true remaining)':>30}")
print("-" * 64)

prev_live = None
for it in range(140):
    selected = [DisallowTracker.Empty, DisallowTracker.Empty]
    order = random.sample(range(2), 2)          # random fill order, like search()
    live0 = None
    for cyc in order:
        selected[cyc] = DisallowTracker.To_Fill
        mask = tr.get_disallowed_selection_mask(selected)
        if cyc == 0:
            live0 = S0 - len(mask)              # <-- the sidebar number this iter
        if cyc == 1:                            # partner: Thompson draw, favours high mu
            draw = mu1 + std1 * np.random.normal(size=S1)
            for m in mask:
                draw[m] = -1e9
            selected[1] = int(np.argmax(draw))
        else:
            choices = [i for i in range(S0) if i not in mask]
            selected[0] = random.choice(choices)
    tr.update(selected)

    # print rows where the live value CHANGED, to make the bouncing obvious
    if prev_live is not None and live0 != prev_live:
        arrow = "UP  ^" if live0 > prev_live else "down v"
        print(f"{it:>4} | {live0:>22} | {ceiling_slot0():>30}   {arrow}")
    prev_live = live0

print("\nCeiling never rose (nothing un-retires); LIVE moved both ways purely from")
print("which partner/fill-order each iteration happened to use. That is your 34870<->34875.")