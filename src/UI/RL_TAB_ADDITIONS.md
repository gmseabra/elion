# Two additions to the 🧪 RL tab

1. **The verdict strip is now a filter.** Click a tile → only the diagnostics
   behind that verdict, lifted above `#tsRlLoop`.
2. **📈 Score distribution.** A new button in the Reactions row pools every TS
   results CSV under a directory and shows the real histogram with a fitted
   normal over it.

`#tsRlLoop` — the closed-loop operations bar — is never moved, hidden, or
re-rendered by either. It is the control surface; everything else is reading
material arranged around it.

---

## 1. Verdict tiles → diagnostic filter

| File | Change |
|---|---|
| `web/static/js/ts/ts_rl.js` | **markup only** — `data-rlv` on the 6 tiles, `data-rlid` + `data-rlcat` on the 9 cards |
| `web/static/js/ts/ts_rl_focus.js` | **new** — the behaviour |
| `web/static/js/hub.js` | +1 loader entry |

**The default is empty.** With no tile selected the tab shows *no* diagnostics
— not all nine. `.grid` is collapsed once at install (`.rl-cardhost`) and never
used as a display surface again; cards are only ever seen inside
`#tsRlFocusHost`. So "nothing selected" and "everything selected" are different
states, and the tab opens on the quiet one.

**And it costs no layout.** `#tsRlFocusHost` is `hidden` at rest, not merely
emptied, so it contributes zero height. `#tsRlLoop` therefore sits exactly where
it did before this file existed — measured, not assumed: the bar's
`getBoundingClientRect().top` is identical (134px in the test fixture) with the
script loaded and with it absent, and returns to that value after a filter is
cleared.

That rules out a helper line in the empty state. An earlier build rendered
*"Pick a verdict above…"* there; it read well and pushed the operations bar down
the page, which is the wrong trade. The affordance lives on the tiles instead —
pointer cursor, hover border, `aria-pressed`, and a card count — where it costs
no layout.

Tiles multi-select (union), show a card count, and carry `aria-pressed`.
Keyboard works (`role="button"`, Enter/Space). "✕ Clear focus" returns to empty.

### The category map

| Tile | Cards |
|---|---|
| Search | 5 (reward-hacker), 6 (uncertainty), wiring |
| Synthesizability | 7 (Suzuki collision), wiring |
| DMTA cycle cost | loop diagram, 4 (n=18), wiring |
| Critic | 1 (no losses), 2 (pose shift), 3 (resolution), wiring |
| Calibration | 3 (resolution), 4 (n=18), wiring |
| Net risk | loop diagram, 5, 6, 7, wiring |

It lives in the **markup**, not in a table inside the JS. A card and its
category are the same line of HTML, so they cannot drift, and adding a tenth
card needs no JS edit.

### Three things that would have gone wrong

**Cards are moved, not cloned.** Four diagnostics reference shared SVG markers
by id — `url(#ar)`, `#arb`, `#arg`. `cloneNode` duplicates those ids and the
arrowheads resolve to whichever copy the document reaches first, so a cloned
card silently steals them from the original. `appendChild` relocates, so there
is exactly one of each id at all times. Same reasoning `_tsRlAdoptPose` already
uses for the pose workspace. Pinned by a test that asserts `cloneNode` never
appears in the file, and by a browser check that `marker#ar` stays unique.

**Restore replays a recorded order, not `nextSibling`.** A stored reference node
can itself be sitting in the focus host by the time you restore. The grid's
original child order is captured once at install and re-appended — deterministic,
and idempotent if it runs twice.

**A backtick in a CSS comment kills the whole file, and `node --check` passes.**
The stylesheets live in template literals. Writing a comment that quotes a
selector in backticks ends the string there, and the rest of the CSS is parsed
as JavaScript — which stays *syntactically valid* (it reads as a tagged
template), so `node --check` is clean and the browser throws
`".grid is not a function"` at load with the feature completely dead. This one
actually happened during the build and was caught only by rendering the page.
G82's shape exactly. Guarded now by two tests: an odd-backtick count across the
four RL files, and a brace-balance check on each extracted stylesheet.

**The empty-tile failure is silent.** A tile whose category matches no card just
opens an empty panel; a card with a typo'd category becomes unreachable. That is
G58's shape exactly, so it gets G58's fix — a cross-reference test
(`tests/test_ts_rl_focus_contract.py`), plus `_tsRlFocusAudit()` in the console
and a live card count on every tile.

---

## 2. 📈 Score distribution

| File | Change |
|---|---|
| `uiapp/routes/ts_routes.py` | **new endpoint** `GET /vina_visualization/ts_score_dist`, + `math` on the import line |
| `web/static/js/ts/ts_rl_loop.js` | the 📈 button, in `#tsRlLoop`'s control bar beside **▶ Run loop** |
| `web/static/js/ts/ts_score_dist.js` | **new** — the overlay and both charts |
| `web/static/js/hub.js` | +1 loader entry |

Point it at `…/RL_active_learning_loop/pre_train` and it walks **recursively**,
so `suzuki/` and `snar/` pool in one pass. `ts_meta` is untouched — it is
non-recursive and has other callers.

The button sits in the loop card's control bar next to **▶ Run loop**;
`ts_warmup.js` is byte-identical to your upload. `ts_score_dist.js` builds its
own popover on first click and appends it to `<body>` — not next to the button,
because `#tsRlLoop` is inside the modal's scroll container and a `position:fixed`
child of a scrolled ancestor is positioned against *that ancestor*, not the
viewport. Owning the popover is also what makes the button relocatable: move the
one line in `ts_rl_loop.js` and nothing is orphaned in a template.

### Two charts, because there are two questions

**Pooled histogram + fitted normal.** The bars are the real scores; the curve is
a normal fitted to them, converted to *expected counts* so it shares one y-axis.
That inversion is the point: a filename that says `mean8.758_std0.475` does not
make the distribution Gaussian. If it is skewed or bimodal the bars visibly
leave the curve. The hover readout gives observed vs expected per bin, and a
`skew` stat tile names it in words.

**Per-run mean ± 1σ, chronological.** Where a biased round shows up — a run
starting from a biased prior should sit visibly off the pooled-mean line.

45 overlaid gaussians was the obvious build and is a spaghetti chart: one hue ×
45 is texture, no run is identifiable, and on a shared count axis each is ~1/45
the height of the pooled curve and simply vanishes.

### The pose step is a prerequisite, not a choice

The scorer dropdown used to list `vina / deepatom / yupu_gign` side by side,
which read as "pick your scorer" and implied you could skip the pose. **DeepAtom
GIGN and GatorAffinity score a ligand-receptor complex; none of them can build one.** Vina is what
produces the 3-D structure they need.

The loop always ran Vina first — only the label was wrong. So the fix is
labelling plus one new control, and the option *values* are unchanged so the
driver is untouched:

```
POSE  [Vina 1.2.7 · always]   RE-SCORE WITH  [ nothing — keep the Vina ΔG ▾ ]
▶ Run Vina search   ▶ Run loop   📈 Score distribution
```

**▶ Run Vina search** sits immediately before ▶ Run loop, so the order encodes
the dependency. It routes through a guard rather than calling
`PoseGen.vina.eng.run()` directly: that function returns silently when there is
no ligand, no receptor or no engine, logging into the *pose panel's* mini-log,
which is a different part of the page. Pressed from this card you would see
nothing happen. Same call, but the reason lands in this card's status line, and
completion reports the affinity.

**Pose preview** — a live 2-D view of the current ligand, in the card. It is a
**copy** of `#poseSvgBox` kept in sync by a `MutationObserver`, not the node
itself: `pose.js` owns that id, and `ts_rl.js` already relocates the whole pose
workspace between `#poseModal` and this tab, so borrowing one child out of it
would strand the 2-D view here the moment the workspace went home.

Copying an `<svg>` is only safe when nothing inside it is addressed by id — a
duplicated `url(#grad)` resolves to whichever copy the document reaches first,
the same collision that nearly cost the diagnostics their arrowheads.
`pose.js`'s 2-D output is pure `<line>`/`<circle>` with inline attributes (zero
`url(#` in the file, verified), and the mirror **re-checks at runtime** and
refuses with a visible reason if that ever stops being true — because it is a
property of `pose.js` today, not a contract it promised.

⚠ **A `let` declared below its first use is a boot-killer.** The observer handle
started as a module-level `let`; `_tsRlLInstall()` runs at boot and reads it,
which is inside its temporal dead zone, so it threw
`Cannot access '_tsRlLPoseObs' before initialization` and took the entire
install down — no card, no loop, no preview. Function *declarations* hoist;
`let` bindings do not. It now lives on the `_tsRlL` state object declared at the
top, and a test forbids the `let` form.

### Drill down and compare

The file list is a **collapsed accordion of top-level folders** — this directory
shows two rows, `snar` and `suzuki`, not 45. Every row is a control:

| | |
|---|---|
| **click** a folder or file | scope both charts to it |
| **📌** | pin it; the next click renders an A/B comparison against it |
| **▸ / ▾** | expand to the individual runs, which are the same two controls |

Comparison mode switches the y-axis from counts to **density**, and says so in
the subtitle. Two selections almost never hold the same number of molecules —
33 files vs 12 here — so overlaid counts would compare sample sizes rather than
shapes. The Δ-mean tile also reports **Cohen's d** against the pooled σ, because
a mean shift is only meaningful relative to the spread it sits in.

The endpoint now returns a **per-file histogram on the same bin grid** as the
pooled one. That is what makes this exact rather than approximate: a subset is
the elementwise sum of its members' counts — no re-read, no re-binning, and two
compared subsets have identical bin edges *by construction*. Binning each subset
on its own range is the classic way to make two distributions look different
when only the binning differs. μ and σ come from the per-file moments, not read
back off 48 bins, which would quantise them to the bin width.

Two things the rendered page caught that no assertion would have: the mean-line
labels sat on top of the run dots (they now carry an opaque chip, because with
two sets interleaved no side of the plot is reliably empty), and the per-run
chart originally *concatenated* the two sets — drawing two campaigns as if one
finished before the other started, a claim about chronology the data does not
make. It now interleaves by timestamp.

### Honesty features

- **Filename drift.** The engine writes `mean…_std…` into the name. The endpoint
  recomputes from the rows and returns `name_drift`. A file edited, appended to,
  or renamed after its run is flagged in red rather than quietly averaged in.
- **Every cap that bites is named** (`caps.files_truncated`, `caps.rows_truncated`)
  and rendered as a banner. A truncated scan never reads as full coverage.
- `_tmp.csv` (a run still writing) is skipped **and listed as skipped**.
- A degenerate zero-variance input widens its own range instead of dividing by
  zero — tested.

### Palette

`#0891b2` pooled, `#8b5cf6` per-run, on surface `#0b1120` — the house trio
already in `ts_rl.js`. Run through the dataviz validator rather than assumed:

```
[PASS] lightness band · [PASS] chroma floor
[PASS] CVD separation  ΔE 12.6 deutan · 10.2 tritan
[PASS] normal-vision   ΔE 21.1
[PASS] contrast vs surface
```

---

## Verify

```bash
python -m pytest tests/ -q          # 212 passed, 17 failed
python -m pytest tests/test_ts_score_dist.py tests/test_ts_rl_focus_contract.py -q   # 38 passed
node --check web/static/js/ts/ts_rl_focus.js web/static/js/ts/ts_score_dist.js
```

**The 17 failures are pre-existing.** Running the suite on the pristine upload
gives the identical 17 (`test_devmap_routes` ×15, `test_excluded_features_absent`,
`test_no_hardcoded_user_paths`); the failure sets `diff` clean. 174 → 212 passing.

Most of those 15 devmap failures trace to `tests/test_elion_vina.py:60`:

```python
sys.modules["uiapp"].app = _test_app
```

which replaces the real app globally at import time for the rest of the session.
Every later test that does `from uiapp import app` gets `<Flask 'tests.test_elion_vina'>`.
Not fixed here — it is outside this change — but it is the cause.

### ⚠ Do not use `pytest.approx` in this suite

It cost an hour, so it is written down. `approx` compares via
`isinstance(value, numpy.ndarray)`, and the stub harnesses replace `numpy` with
a `MagicMock` — at which point `numpy.ndarray` is a Mock attribute, not a type:

```
TypeError: isinstance() arg 2 must be a type, a tuple of types, or a union
```

It **passes when the file runs alone and fails inside the full suite**, which
reads as a flaky test rather than a stubbed dependency. `_close()` in
`tests/test_ts_score_dist.py` is a three-line replacement with no hidden import.

### Not updated

`tests/test_import_smoke.py` — one more route was added (`ts_score_dist`). Its
endpoint inventory is a whitelist rather than a count, so nothing failed, but
add the endpoint there if you want it pinned.
