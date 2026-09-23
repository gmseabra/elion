#!/usr/bin/env python
# =============================================================================
# uiapp/core/rl_bias.py — the RL feedback algorithm. ONE implementation.
#
# WHAT THIS IS
# ------------
# The closed loop is:
#
#     TS run  →  top 30% by ChemBERT score  →  Vina pose  →  DeepAtom/GIGN pK
#             →  THIS FILE  →  warm-up checkpoint  →  next TS run starts biased
#
# This module owns the two steps that must agree everywhere:
#
#   1. select_top_fraction()  — which molecules get posed and predicted.
#      Called by /ts_rl_harvest (to build the browser's work list) AND by the
#      elion-side bias_generator (when it does the selection itself). If these
#      two ever disagreed, the UI would pose one set and the engine would
#      credit another, and nothing would report the mismatch.
#
#   2. bias_from_affinity()   — measured pK  →  per-reagent posterior nudge,
#      MERGED INTO the previous checkpoint and written as a new one.
#
# WHY IT IS DEPENDENCY-FREE
# -------------------------
# stdlib only (json, math, os, glob, csv, argparse). No numpy, no rdkit, no
# torch. That is deliberate: this file is imported in-process by the Flask
# route AND loaded by absolute path from the elion tree (which runs in a
# different conda env). A shared file with no dependencies cannot fail to
# import on one side and work on the other.
#
# THE TRAP THIS FILE EXISTS TO AVOID
# ----------------------------------
# The elion-side loader (written by ts_routes._write_warmup_loader) restores a
# reagent's belief ONLY if its reagent_name appears in the checkpoint's
# `components` map. Every reagent NOT in the file is reset to the global
# prior_mean/prior_std.
#
# So writing a checkpoint containing only the ~60 reagents you just measured
# would silently discard the learned posteriors of the other ~80,000 — a full
# run's worth of Bayesian updating, destroyed by a file that looks correct and
# loads without error. `merge_into_checkpoint()` therefore REQUIRES a previous
# checkpoint to carry forward, and refuses to write without one unless the
# caller passes allow_no_prior=True and accepts what that means.
# =============================================================================

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time

SCHEMA = "elion.rl.affinity/1"
IMPL_VERSION = "1.0.0"

# pK  =  -dG / 1.36   — the same constant pose_routes.py:958 and
# deepatom_routes.py:63 use. Kept here so all four call sites agree.
KCAL_PER_LOG = 1.36

DEFAULT_FRACTION = 0.30      # "top 30% of the ChemBERT estimated affinity"
DEFAULT_STRENGTH = 0.5       # how far a measurement may move a posterior mean
# new_std is not allowed BELOW this × prior_std. It must be small: a converged
# reagent's std is prior_std/sqrt(n+1), so after 9 observations it is already
# 0.32 × prior_std. A floor anywhere near that would RAISE the uncertainty of
# the reagents TS is most confident about — a feedback round un-learning what
# the run learned. See merge_into_checkpoint for the clamp that makes the floor
# a lower bound only, never a lift.
DEFAULT_STD_FLOOR = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# 1. Selection — the "top 30%" rule
# ─────────────────────────────────────────────────────────────────────────────

def read_results_csv(path: str) -> list:
    """Read a TS results CSV into [{score, smiles, name}].

    The engine writes `score,SMILES,Name` (ts_main.run_ts). Rows whose score
    does not parse as a float are dropped and counted, not silently skipped —
    the count comes back in select_top_fraction's stats.
    """
    rows, bad = [], 0
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        # Column names have drifted across scripts (score/Score/LABELS,
        # SMILES/smiles). Resolve once, by lowercased header, rather than
        # assuming one spelling.
        cols = {(c or "").strip().lower(): c for c in (rdr.fieldnames or [])}
        c_score = cols.get("score") or cols.get("labels") or cols.get("affinity")
        c_smi = cols.get("smiles")
        c_name = cols.get("name") or cols.get("id")
        if not c_score or not c_smi:
            raise ValueError(
                "results CSV %s has no score/SMILES columns (found: %s)"
                % (path, rdr.fieldnames))
        for r in rdr:
            try:
                sc = float(r[c_score])
            except (TypeError, ValueError):
                bad += 1
                continue
            if not math.isfinite(sc):
                bad += 1
                continue
            rows.append({
                "score": sc,
                "smiles": (r.get(c_smi) or "").strip(),
                "name": (r.get(c_name) or "").strip() if c_name else "",
            })
    read_results_csv.last_unparseable = bad
    return rows


def select_top_fraction(rows: list, fraction: float = DEFAULT_FRACTION,
                        cap: int | None = None, mode: str = "maximize") -> tuple:
    """Top `fraction` of rows by score, deduplicated by SMILES.

    Returns (selected, stats). `stats` is the honest accounting — it always
    reports n_unique, n_fraction and whether the cap bit, so a caller can say
    "posing 30 of 1187 (top 30%, capped)" instead of presenting a truncated
    list as if it were the whole selection.

    Dedup keeps the BEST-scoring occurrence of each SMILES, not the first: TS
    revisits products, and the first visit is not necessarily the best-scored
    one. Deduplication is on the raw SMILES string — the engine emits
    Chem.MolToSmiles output, which is already canonical, so canonicalising
    again here would cost an rdkit dependency for no gain.
    """
    reverse = (mode != "minimize")
    best: dict[str, dict] = {}
    blank = 0
    for r in rows:
        smi = r.get("smiles") or ""
        if not smi:
            blank += 1
            continue
        prev = best.get(smi)
        if prev is None or (r["score"] > prev["score"]) == reverse:
            best[smi] = r

    uniq = sorted(best.values(), key=lambda r: r["score"], reverse=reverse)
    frac = max(0.0, min(1.0, float(fraction)))
    n_frac = max(1, int(round(len(uniq) * frac))) if uniq else 0
    picked = uniq[:n_frac]
    capped = False
    if cap is not None and cap > 0 and len(picked) > cap:
        picked = picked[:cap]
        capped = True

    stats = {
        "n_rows": len(rows),
        "n_blank_smiles": blank,
        "n_unique": len(uniq),
        "fraction": frac,
        "n_fraction": n_frac,
        "n_selected": len(picked),
        "cap": cap,
        "capped": capped,
        "mode": mode,
        "score_min": (min(r["score"] for r in picked) if picked else None),
        "score_max": (max(r["score"] for r in picked) if picked else None),
    }
    return picked, stats


# ─────────────────────────────────────────────────────────────────────────────
# 2. Credit assignment — product measurement  →  per-reagent evidence
# ─────────────────────────────────────────────────────────────────────────────

def split_product_name(name: str) -> list:
    """"<ridA>_<ridB>" → ["ridA", "ridB"], else [].

    thompson_sampling.evaluate() builds the product name as
    "_".join(r.reagent_name for r in selected_reagents), IN COMPONENT ORDER —
    so position in this list IS the component index. That is the only reason
    a measured product can be credited to the right slot.
    """
    parts = str(name or "").replace("+", "_").replace("|", "_").split("_")
    parts = [p for p in parts if p]
    return parts if len(parts) >= 2 else []


def record_pk(rec: dict) -> float | None:
    """The measured affinity of one record as a pK (higher = better).

    Accepts either `pk` (DeepAtom/GIGN native) or `dg` in kcal/mol (the
    in-browser Vina score), converting the latter. Returns None when the
    record carries neither, or is flagged not-ok — a failed scorer must not
    enter the prior as a zero.
    """
    if not rec.get("ok", True):
        return None
    for key in ("pk", "pred_pk"):
        v = rec.get(key)
        if isinstance(v, (int, float)) and math.isfinite(v):
            return float(v)
    for key in ("dg", "deltaG", "vina_dg"):
        v = rec.get(key)
        if isinstance(v, (int, float)) and math.isfinite(v) and v < 0:
            return -float(v) / KCAL_PER_LOG
    return None


def reagent_evidence(records: list) -> tuple:
    """Collapse product measurements into per-reagent evidence.

    Mirrors thompson_sampling._flush_score_batch: a product's score is credited
    to EVERY reagent that contributed to it. A reagent measured in several
    products gets the mean of those measurements.

    Returns (evidence, stats) where evidence is
        {reagent_name: {"pk": mean_pk, "n": count, "comp": component_index}}
    """
    acc: dict[str, dict] = {}
    used = skipped = unnamed = 0
    for rec in records:
        pk = record_pk(rec)
        if pk is None:
            skipped += 1
            continue
        names = rec.get("reagents") or split_product_name(rec.get("name", ""))
        if not names:
            unnamed += 1
            continue
        used += 1
        for comp, rname in enumerate(names):
            e = acc.setdefault(str(rname), {"sum": 0.0, "n": 0, "comp": comp})
            e["sum"] += pk
            e["n"] += 1
    evidence = {k: {"pk": v["sum"] / v["n"], "n": v["n"], "comp": v["comp"]}
                for k, v in acc.items()}
    stats = {
        "n_records": len(records),
        "n_used": used,
        "n_no_score": skipped,
        "n_unresolved_name": unnamed,
        "n_reagents": len(evidence),
    }
    return evidence, stats


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mapping — measured pK  →  a move on the reward scale
# ─────────────────────────────────────────────────────────────────────────────

def _mean_std(xs: list) -> tuple:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / n
    return m, math.sqrt(var)


def compute_shifts(evidence: dict, prior_std: float, mapping: str = "zblend",
                   strength: float = DEFAULT_STRENGTH) -> tuple:
    """Turn per-reagent pK evidence into an additive shift on the reward scale.

    THE UNITS PROBLEM, stated plainly: TS maximises the Elion reward
    (CHEMBERT_BE at rew_coeff 0.95, plus SAScore/QED). The pose critics return
    a pK. These are different quantities on different scales, and writing one
    into a field holding the other is the single easiest way to make this loop
    quietly useless.

    'zblend' (default) sidesteps it by never using the pK's absolute value —
    only its position within the batch:

        z_i   = (pk_i - mean(pk)) / std(pk)
        shift = z_i * prior_std * strength

    prior_std is the spread of the reward TS actually saw, so the shift is
    expressed in the units of the thing being biased. strength is the only
    tuning knob and it means what it says: at 0.5, a reagent one standard
    deviation better than the batch moves half a reward-sigma.

    'raw'  writes the pK straight in. Returned with a loud warning because the
           scales do not match; kept because it is the right thing when TS is
           itself optimising a pK-like objective.
    'rank' ignores magnitudes entirely: top third up, bottom third down, by a
           fixed ±0.5·prior_std·strength. Most robust to a miscalibrated critic.
    """
    warnings = []
    names = sorted(evidence)
    pks = [evidence[n]["pk"] for n in names]
    if not names:
        return {}, {"mapping": mapping, "warnings": ["no evidence"], "n": 0}

    mean_pk, std_pk = _mean_std(pks)
    shifts: dict[str, float] = {}

    if mapping == "raw":
        warnings.append(
            "mapping='raw': pK is written directly into current_mean. TS is "
            "optimising the Elion reward, which is on a different scale — the "
            "injected priors will be either ignored or dominant, not blended.")
        for n in names:
            shifts[n] = None            # absolute, handled by the merge step
    elif mapping == "rank":
        order = sorted(names, key=lambda n: evidence[n]["pk"], reverse=True)
        third = max(1, len(order) // 3)
        step = 0.5 * prior_std * strength
        for i, n in enumerate(order):
            shifts[n] = step if i < third else (-step if i >= len(order) - third else 0.0)
    else:                                # zblend
        if std_pk < 1e-9:
            warnings.append(
                "every measured pK is identical (std=%.3g) — z is undefined, so "
                "no reagent is moved. This is the honest outcome: a critic that "
                "cannot separate the batch carries no information about it."
                % std_pk)
            for n in names:
                shifts[n] = 0.0
        else:
            for n in names:
                z = (evidence[n]["pk"] - mean_pk) / std_pk
                shifts[n] = z * prior_std * strength

    stats = {
        "mapping": mapping,
        "strength": strength,
        "prior_std": prior_std,
        "pk_mean": round(mean_pk, 4),
        "pk_std": round(std_pk, 4),
        "pk_min": round(min(pks), 4),
        "pk_max": round(max(pks), 4),
        "n": len(names),
        "warnings": warnings,
    }
    return shifts, stats


# ─────────────────────────────────────────────────────────────────────────────
# 4. Checkpoint I/O — merge, never replace
# ─────────────────────────────────────────────────────────────────────────────

def latest_checkpoint(warmup_dir: str, short_name: str) -> str:
    """Newest <short_name>_*_warmup.json, or "".

    Newest BY NAME, reverse-sorted — the same rule ts_routes._warmup_cache_path
    uses to decide which checkpoint the next run loads. Matching that rule is
    the whole point: if this picked by mtime and that picked by name, the file
    we merged into would not be the file the engine reads.
    """
    matches = sorted(
        (p for p in glob.glob(os.path.join(warmup_dir, "%s_*_warmup.json" % short_name))
         if os.path.getsize(p) > 0),
        reverse=True)
    return matches[0] if matches else ""


def load_checkpoint(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def merge_into_checkpoint(ckpt: dict, evidence: dict, shifts: dict,
                          strength: float = DEFAULT_STRENGTH,
                          std_floor: float = DEFAULT_STD_FLOOR,
                          allow_no_prior: bool = False) -> tuple:
    """Apply the shifts to a COPY of `ckpt`, leaving every other reagent alone.

    Returns (new_ckpt, report).

    Three things happen to a measured reagent and nothing happens to any other:
      · current_mean += shift            (or = pk, under mapping='raw')
      · current_std  clamped — see below
      · num_scores   += the number of products it was measured in

    THE STD CLAMP, because getting it wrong is silent and backwards:

        new_std = min(old_std, max(old_std * shrink, floor))

    current_std is what the Thompson draw samples with, so it IS the
    exploration budget for that reagent. Two things must both hold, and the
    obvious `max(old_std * shrink, floor)` only gets one of them:

      · it must not collapse. One docking campaign is worth about one
        observation of confidence — that is what 1/sqrt(1+strength) encodes —
        and `floor` stops repeated rounds compounding toward zero.
      · it must not RISE. A converged reagent's std is prior_std/sqrt(n+1),
        already 0.32 × prior_std after nine observations. A bare max() against
        any floor above that hands the reagent MORE uncertainty than it had —
        so the round that measured your best building block would make the
        bandit less sure of it. The outer min(old_std, …) forbids that: the
        floor is a lower bound on how far we will tighten, never a licence to
        loosen.
    """
    if not ckpt:
        if not allow_no_prior:
            raise ValueError(
                "no previous warm-up checkpoint to merge into. Writing one now "
                "would contain ONLY the reagents just measured, and the engine "
                "loader resets every reagent absent from the file to the global "
                "prior — silently discarding the whole run's learned posteriors. "
                "Run TS once to completion first (it writes the checkpoint from "
                "its own warm-up), or pass allow_no_prior=True if you genuinely "
                "want a measurement-only prior.")
        ckpt = {"prior_mean": 0.0, "prior_std": 1.0, "components": {}}

    prior_mean = float(ckpt.get("prior_mean") or 0.0)
    prior_std = float(ckpt.get("prior_std") or 1.0)
    known_var = float(ckpt.get("known_var") or (prior_std ** 2))
    floor = std_floor * prior_std
    shrink = 1.0 / math.sqrt(1.0 + max(0.0, strength))

    out = {
        "rxn_key": ckpt.get("rxn_key", ""),
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "known_var": known_var,
        "components": {},
    }

    applied, missing = [], []
    seen: set = set()
    n_carried = 0

    for comp_key, comp_list in (ckpt.get("components") or {}).items():
        new_list = []
        for r in comp_list:
            rname = str(r.get("reagent_name", ""))
            nr = dict(r)
            if rname in evidence:
                seen.add(rname)
                sh = shifts.get(rname)
                old_mean = float(nr.get("current_mean", prior_mean))
                old_std = float(nr.get("current_std", prior_std))
                if sh is None:                     # mapping='raw'
                    new_mean = evidence[rname]["pk"]
                else:
                    new_mean = old_mean + sh
                new_std = min(old_std, max(old_std * shrink, floor))
                nr["current_mean"] = round(new_mean, 6)
                nr["current_std"] = round(new_std, 6)
                nr["num_scores"] = int(nr.get("num_scores", 0)) + evidence[rname]["n"]
                nr["known_var"] = float(nr.get("known_var") or known_var)
                nr["rl_shift"] = None if sh is None else round(sh, 6)
                nr["rl_pk"] = round(evidence[rname]["pk"], 4)
                applied.append({
                    "reagent_name": rname, "component": comp_key,
                    "mean_before": round(old_mean, 6), "mean_after": round(new_mean, 6),
                    "std_before": round(old_std, 6), "std_after": round(new_std, 6),
                    "pk": round(evidence[rname]["pk"], 4), "n": evidence[rname]["n"],
                })
            else:
                n_carried += 1
            new_list.append(nr)
        out["components"][str(comp_key)] = new_list

    # Measured reagents the checkpoint has never heard of. They are ADDED at
    # their split-derived component with the global prior as a starting point —
    # but they are also reported, because a large `missing` count means the
    # checkpoint and the results CSV came from different runs.
    for rname, ev in evidence.items():
        if rname in seen:
            continue
        comp = str(ev.get("comp", 0))
        sh = shifts.get(rname)
        new_mean = ev["pk"] if sh is None else (prior_mean + (sh or 0.0))
        out["components"].setdefault(comp, []).append({
            "reagent_name": rname,
            "current_mean": round(new_mean, 6),
            "current_std": round(min(prior_std, max(prior_std * shrink, floor)), 6),
            "known_var": known_var,
            "num_scores": ev["n"],
            "rl_shift": None if sh is None else round(sh, 6),
            "rl_pk": round(ev["pk"], 4),
        })
        missing.append(rname)

    out["n_components"] = len(out["components"])
    out["n_reagents"] = sum(len(v) for v in out["components"].values())

    report = {
        "n_applied": len(applied),
        "n_added": len(missing),
        "n_carried_unchanged": n_carried,
        "added_reagents": missing[:25],
        "added_truncated": max(0, len(missing) - 25),
        "applied": applied[:50],
        "applied_truncated": max(0, len(applied) - 50),
        "prior_mean": prior_mean,
        "prior_std": prior_std,
        "std_floor": round(floor, 6),
        "std_shrink": round(shrink, 6),
    }
    return out, report


def write_checkpoint(ckpt: dict, warmup_dir: str, short_name: str,
                     timestamp: str = "") -> str:
    """Write a new timestamped checkpoint. Atomic (tmp + os.replace).

    The filename must sort ABOVE every existing one, because both this module
    and ts_routes._warmup_cache_path select by reverse name sort. A
    %Y%m%d_%H%M%S stamp does that for any run after the last one.
    """
    os.makedirs(warmup_dir, exist_ok=True)
    ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
    ckpt["timestamp"] = ts
    ckpt["rl_biased"] = True
    ckpt["rl_impl_version"] = IMPL_VERSION
    path = os.path.join(warmup_dir, "%s_%s_warmup.json" % (short_name, ts))
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(ckpt, fh, indent=1)
    os.replace(tmp, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 5. The one call the outside world makes
# ─────────────────────────────────────────────────────────────────────────────

def bias_from_affinity(affinity_path: str, warmup_dir: str,
                       short_name: str = "", mapping: str = "zblend",
                       strength: float = DEFAULT_STRENGTH,
                       std_floor: float = DEFAULT_STD_FLOOR,
                       allow_no_prior: bool = False,
                       dry_run: bool = False) -> dict:
    """affinity JSON  →  a new warm-up checkpoint. Returns a full report.

    Nothing here is silent: every count that could hide a problem (records
    with no score, names that would not split, reagents the checkpoint did not
    know) is in the returned dict, and the caller is expected to show them.
    """
    with open(affinity_path) as fh:
        aff = json.load(fh)

    if aff.get("schema") != SCHEMA:
        raise ValueError("affinity file %s has schema %r, expected %r"
                         % (affinity_path, aff.get("schema"), SCHEMA))

    short = short_name or aff.get("short_name") or aff.get("rxn_key") or "ts"
    records = aff.get("records") or []

    evidence, ev_stats = reagent_evidence(records)
    if not evidence:
        return {
            "ok": False,
            "err": "no usable measurements in %s (%d records, %d without a "
                   "score, %d whose name would not split into reagent ids)"
                   % (affinity_path, ev_stats["n_records"],
                      ev_stats["n_no_score"], ev_stats["n_unresolved_name"]),
            "evidence": ev_stats,
        }

    prev_path = latest_checkpoint(warmup_dir, short)
    prev = load_checkpoint(prev_path) if prev_path else {}
    prior_std = float((prev or {}).get("prior_std") or 1.0)

    shifts, map_stats = compute_shifts(evidence, prior_std, mapping, strength)
    new_ckpt, report = merge_into_checkpoint(
        prev, evidence, shifts, strength=strength, std_floor=std_floor,
        allow_no_prior=allow_no_prior)
    new_ckpt["rxn_key"] = new_ckpt.get("rxn_key") or aff.get("rxn_key", "")
    new_ckpt["rl_source_affinity"] = os.path.abspath(affinity_path)
    new_ckpt["rl_prev_checkpoint"] = prev_path

    out_path = "" if dry_run else write_checkpoint(new_ckpt, warmup_dir, short)

    return {
        "ok": True,
        "dry_run": dry_run,
        "checkpoint": out_path,
        "prev_checkpoint": prev_path,
        "short_name": short,
        "rxn_key": new_ckpt.get("rxn_key", ""),
        "scorer": aff.get("scorer", ""),
        "selection": aff.get("selection", {}),
        "evidence": ev_stats,
        "mapping": map_stats,
        "merge": report,
        "impl_version": IMPL_VERSION,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLI — how the elion side runs it
# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Bias a Thompson-Sampling reagent prior from measured "
                    "pose affinities. Writes a warm-up checkpoint the next TS "
                    "run loads automatically.")
    ap.add_argument("--affinity", required=True, help="affinity JSON from the UI")
    ap.add_argument("--warmup-dir", required=True, help="the Warmup_TS directory")
    ap.add_argument("--short-name", default="", help="reaction short name (default: from the file)")
    ap.add_argument("--mapping", default="zblend", choices=["zblend", "raw", "rank"])
    ap.add_argument("--strength", type=float, default=DEFAULT_STRENGTH)
    ap.add_argument("--std-floor", type=float, default=DEFAULT_STD_FLOOR)
    ap.add_argument("--allow-no-prior", action="store_true",
                    help="write a checkpoint even with no previous one to merge "
                         "into. Read merge_into_checkpoint's docstring first.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    a = ap.parse_args(argv)

    try:
        rep = bias_from_affinity(
            a.affinity, a.warmup_dir, a.short_name, a.mapping, a.strength,
            a.std_floor, a.allow_no_prior, a.dry_run)
    except Exception as exc:
        print("RL_BIAS_ERROR %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2

    if a.json:
        print(json.dumps(rep, indent=1))
        return 0 if rep.get("ok") else 1

    if not rep.get("ok"):
        print("RL_BIAS_ERROR " + rep.get("err", "unknown"), file=sys.stderr)
        return 1

    m, mg, ev = rep["mapping"], rep["merge"], rep["evidence"]
    print("[rl_bias] %s  mapping=%s strength=%.2f" % (rep["short_name"], m["mapping"], m["strength"]))
    print("[rl_bias] measurements: %d records -> %d usable -> %d reagents "
          "(%d no score, %d unresolved name)"
          % (ev["n_records"], ev["n_used"], ev["n_reagents"],
             ev["n_no_score"], ev["n_unresolved_name"]))
    print("[rl_bias] pK  mean=%.3f std=%.3f  range [%.3f, %.3f]"
          % (m["pk_mean"], m["pk_std"], m["pk_min"], m["pk_max"]))
    print("[rl_bias] prior_std=%.4f  std_floor=%.4f  shrink=%.4f"
          % (mg["prior_std"], mg["std_floor"], mg["std_shrink"]))
    print("[rl_bias] applied=%d  added=%d  carried unchanged=%d"
          % (mg["n_applied"], mg["n_added"], mg["n_carried_unchanged"]))
    for w in m.get("warnings", []):
        print("[rl_bias] WARNING: " + w)
    if mg["n_added"]:
        print("[rl_bias] NOTE: %d measured reagents were absent from %s — check "
              "the affinity file and the checkpoint came from the same run."
              % (mg["n_added"], rep["prev_checkpoint"] or "(none)"))
    print("[rl_bias] prev checkpoint: %s" % (rep["prev_checkpoint"] or "(none)"))
    print("[rl_bias] RL_BIAS_CHECKPOINT %s" % (rep["checkpoint"] or "(dry run)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
