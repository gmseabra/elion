#!/usr/bin/env python
# =============================================================================
# elion_bias_generator_TS.py — the elion side of the RL loop.
#
# WHERE THIS GOES
# ---------------
#   cp elion_bias_generator_TS.py  <elion>/src/elion/generators/TS/rl.py
#
# then add ONE method to whatever class Generator('TS') returns:
#
#     from generators.TS.rl import bias_generator as _rl_bias
#
#     class TSGenerator:
#         ...
#         def bias_generator(self, control_cfg, estimator):
#             return _rl_bias(control_cfg, estimator)
#
# and set `run_type: bias_generator` in input_TS.yml. elion.py's existing
# dispatch does the rest — it already calls
# `generator.bias_generator(config['Control'], estimator)`.
#
# WHY IT IS A SHIM AND NOT AN IMPLEMENTATION
# ------------------------------------------
# The arithmetic lives in ONE file, `uiapp/core/rl_bias.py`, in the UI tree.
# Both the Flask route and this module load that same file — the route by
# import, this module by absolute path. Copying the algorithm into the elion
# tree instead would give two copies that agree today and drift the first time
# either is touched, and the failure mode is the worst kind: the UI shows a
# preview computed one way and the engine writes a checkpoint computed the
# other, with nothing comparing them.
#
# rl_bias.py is stdlib-only for exactly this reason — it has to import cleanly
# in the elion conda env, which has no Flask and need not have anything else.
#
# HOW IT FINDS rl_bias.py, in order:
#   1. $ELION_RL_IMPL            — an explicit path to rl_bias.py
#   2. $ELION_UI_ROOT/uiapp/core/rl_bias.py
#   3. a short probe of sibling layouts next to the elion checkout
# If none hit, it raises with the three paths it tried. A silent fallback to a
# vendored copy is precisely the drift this design is avoiding.
#
# WHAT IT DOES
# ------------
#   affinity JSON (written by the UI's /ts_rl_feedback)
#        -> per-reagent evidence  (a product's pK credited to both reagents)
#        -> a shift on the reward scale
#        -> MERGED into the newest Warmup_TS checkpoint
#        -> a new checkpoint the next TS run loads via TS_WARMUP_CHECKPOINT
#
# It prints `RL_BIAS_CHECKPOINT <path>` on success. The UI route parses that
# line, so keep it.
#
# If no affinity file is configured it falls back to SELECTION-ONLY mode: it
# reads the run's results CSV, applies the same top-30% rule, and writes the
# work list to disk for the UI to pose. That is the mode to use if you ever
# want the engine, not the browser, to decide the batch.
# =============================================================================

from __future__ import annotations

import importlib.util
import json
import os
import sys

_TRIED: list = []


def _load_rl_bias():
    """Import uiapp/core/rl_bias.py by absolute path. One implementation."""
    cands = []

    explicit = os.environ.get("ELION_RL_IMPL", "").strip()
    if explicit:
        cands.append(explicit)

    ui_root = os.environ.get("ELION_UI_ROOT", "").strip()
    if ui_root:
        cands.append(os.path.join(ui_root, "uiapp", "core", "rl_bias.py"))

    # Sibling-layout probe. The UI checkout usually sits next to the elion one;
    # this file is at <elion>/src/elion/generators/TS/rl.py, so the elion root
    # is five levels up. Probing beats hardcoding one layout — the app's own
    # ELION_CWD resolution learned that (see the playbook, G92).
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(3, 8):
        root = os.path.abspath(os.path.join(here, *([".."] * up)))
        for name in ("UI", "visualizer", "elion_ui", "ui"):
            cands.append(os.path.join(root, name, "uiapp", "core", "rl_bias.py"))

    for p in cands:
        _TRIED.append(p)
        if p and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("elion_rl_bias", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print("[rl] implementation: %s (v%s)" % (p, getattr(mod, "IMPL_VERSION", "?")),
                  flush=True)
            return mod

    raise ImportError(
        "could not locate uiapp/core/rl_bias.py — the RL feedback algorithm.\n"
        "Set ELION_RL_IMPL to its full path, or ELION_UI_ROOT to the UI "
        "checkout root.\nTried:\n  " + "\n  ".join(_TRIED[:12]))


def _cfg(control_cfg: dict, key: str, env: str, default=None):
    """Read a setting from Control:, then the environment, then the default.

    The env vars are how the UI route passes settings when it launches
    `elion.py -i <patched yml>` — the yml is a copy of the user's, so the route
    must not have to rewrite arbitrary keys into it.
    """
    if isinstance(control_cfg, dict):
        for k in (key, key.replace("_", ""), "rl_" + key):
            if k in control_cfg and control_cfg[k] not in (None, ""):
                return control_cfg[k]
    v = os.environ.get(env, "")
    return v if v not in (None, "") else default


def bias_generator(control_cfg=None, estimator=None):
    """Bias the TS reagent prior from measured pose affinities.

    Signature matches what elion.py hands a generator:
        generator.bias_generator(config['Control'], estimator)

    `estimator` is accepted and DELIBERATELY UNUSED. The mapping from measured
    pK to a reward-scale shift is arithmetic over the previous checkpoint's
    prior_std — it needs no forward pass, so this step does not load ChemBERT
    and does not touch the GPU. If you later want the reward re-evaluated here,
    that is a new mode, not a change to this one: keep the cheap path cheap so
    a feedback commit stays seconds rather than minutes.

    Returns the report dict. Raises on a genuine misconfiguration rather than
    returning a falsy value, so `run_type: bias_generator` fails loudly.
    """
    control_cfg = control_cfg or {}
    rlb = _load_rl_bias()

    warmup_dir = _cfg(control_cfg, "warmup_dir", "ELION_RL_WARMUP_DIR", "")
    affinity = _cfg(control_cfg, "affinity_file", "ELION_RL_AFFINITY", "")
    mapping = _cfg(control_cfg, "mapping", "ELION_RL_MAPPING", "zblend")
    strength = float(_cfg(control_cfg, "strength", "ELION_RL_STRENGTH", rlb.DEFAULT_STRENGTH))
    std_floor = float(_cfg(control_cfg, "std_floor", "ELION_RL_STD_FLOOR", rlb.DEFAULT_STD_FLOOR))
    allow_np = str(_cfg(control_cfg, "allow_no_prior", "ELION_RL_ALLOW_NO_PRIOR", "0")) in ("1", "true", "True")
    dry_run = str(_cfg(control_cfg, "dry_run", "ELION_RL_DRY_RUN", "0")) in ("1", "true", "True")

    if not warmup_dir:
        raise ValueError(
            "no warm-up directory. Set Control.warmup_dir in input_TS.yml or "
            "ELION_RL_WARMUP_DIR in the environment. It is the Warmup_TS folder "
            "under visualizer.output_dir — the same one TS_WARMUP_CHECKPOINT is "
            "read from, or the checkpoint this writes will never be loaded.")

    # ── Selection-only mode: no measurements yet, just pick the batch ──────
    if not affinity:
        results_csv = _cfg(control_cfg, "results_csv", "ELION_RL_RESULTS_CSV", "")
        if not results_csv:
            raise ValueError(
                "bias_generator has nothing to work from. Either set "
                "ELION_RL_AFFINITY to an affinity JSON (measured pose scores), "
                "or ELION_RL_RESULTS_CSV to a finished run's results file to "
                "emit a selection-only work list.")
        frac = float(_cfg(control_cfg, "fraction", "ELION_RL_FRACTION", rlb.DEFAULT_FRACTION))
        cap_raw = _cfg(control_cfg, "cap", "ELION_RL_CAP", "")
        cap = int(cap_raw) if str(cap_raw).strip() else None
        rows = rlb.read_results_csv(results_csv)
        picked, stats = rlb.select_top_fraction(rows, frac, cap)
        out = os.path.join(warmup_dir, "rl_worklist.json")
        os.makedirs(warmup_dir, exist_ok=True)
        with open(out, "w") as fh:
            json.dump({"schema": "elion.rl.worklist/1", "source_csv": results_csv,
                       "selection": stats,
                       "candidates": [{"rank": i + 1, "smiles": r["smiles"], "name": r["name"],
                                       "reagents": rlb.split_product_name(r["name"]),
                                       "ts_score": r["score"]}
                                      for i, r in enumerate(picked)]}, fh, indent=1)
        print("[rl] selection-only: %d unique products -> top %.0f%% = %d%s"
              % (stats["n_unique"], stats["fraction"] * 100, stats["n_selected"],
                 " (capped)" if stats["capped"] else ""), flush=True)
        print("[rl] RL_BIAS_WORKLIST %s" % out, flush=True)
        return {"ok": True, "mode": "selection", "worklist": out, "selection": stats}

    # ── Normal mode: measurements in, biased checkpoint out ────────────────
    rep = rlb.bias_from_affinity(
        affinity, warmup_dir, short_name=str(_cfg(control_cfg, "short_name", "ELION_RL_SHORT", "") or ""),
        mapping=mapping, strength=strength, std_floor=std_floor,
        allow_no_prior=allow_np, dry_run=dry_run)

    if not rep.get("ok"):
        raise RuntimeError("bias_generator: " + rep.get("err", "unknown failure"))

    m, mg, ev = rep["mapping"], rep["merge"], rep["evidence"]
    print("[rl] %s  mapping=%s strength=%.2f" % (rep["short_name"], m["mapping"], m["strength"]), flush=True)
    print("[rl] %d records -> %d usable -> %d reagents (%d unscored, %d unresolved names)"
          % (ev["n_records"], ev["n_used"], ev["n_reagents"],
             ev["n_no_score"], ev["n_unresolved_name"]), flush=True)
    print("[rl] pK mean=%.3f std=%.3f range [%.3f, %.3f]"
          % (m["pk_mean"], m["pk_std"], m["pk_min"], m["pk_max"]), flush=True)
    print("[rl] applied=%d added=%d carried unchanged=%d  (prior sigma %.4f, floor %.4f)"
          % (mg["n_applied"], mg["n_added"], mg["n_carried_unchanged"],
             mg["prior_std"], mg["std_floor"]), flush=True)
    for w in m.get("warnings", []):
        print("[rl] WARNING: %s" % w, flush=True)
    if mg["n_added"]:
        print("[rl] NOTE: %d measured reagents were absent from the previous checkpoint "
              "(%s) — verify both came from the same run."
              % (mg["n_added"], rep["prev_checkpoint"] or "none"), flush=True)

    # ── The line the UI route greps for. Keep it. ──────────────────────────
    print("[rl] RL_BIAS_CHECKPOINT %s" % (rep["checkpoint"] or "(dry run)"), flush=True)
    return rep


# Alias, in case your Generator base class expects the elion.py-level name.
bias = bias_generator


if __name__ == "__main__":
    try:
        r = bias_generator({}, None)
    except Exception as exc:
        print("RL_BIAS_ERROR %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(0 if r.get("ok") else 1)
