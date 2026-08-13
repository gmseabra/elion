#!/usr/bin/env python
"""Contract test for the RL feedback loop. Dependency-free, run it directly:

    python tests_rl_roundtrip.py

It checks the four things that can silently break this loop:

  1. select_top_fraction really is "top 30% by score, deduped, honestly capped"
  2. a product's pK is credited to BOTH of its reagents
  3. the checkpoint rl_bias writes is consumable by the loader ts_routes
     generates — field for field — and every UNMEASURED reagent survives
     the merge unchanged (the destructive-reset trap)
  4. writing with no previous checkpoint is REFUSED by default

Test 3 also greps ts_routes.py for the loader's actual field accesses, so if
someone changes the loader's schema this test fails instead of the next run
silently losing 80,000 posteriors.
"""
import json
import os
import re
import tempfile

# Load rl_bias BY PATH, not as uiapp.core.rl_bias. Importing the package would
# execute uiapp/__init__.py, which imports every route module, which imports
# torch — so a stdlib-only algorithm would become untestable without the ML
# stack. This is also exactly how the elion-side shim loads it, so the test
# exercises the real loading path rather than a convenience one.
import importlib.util                                     # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "rl_bias", os.path.join(_HERE, "uiapp", "core", "rl_bias.py"))
R = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(R)

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print("  ok   %s" % name)
    else:
        print("  FAIL %s  %s" % (name, detail))
        FAILED.append(name)


# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] selection — top 30%, deduped, capped honestly")

with tempfile.TemporaryDirectory() as td:
    csv_path = os.path.join(td, "suzuki_20260810_120000_mean5.100_std0.900.csv")
    with open(csv_path, "w") as fh:
        fh.write("score,SMILES,Name\n")
        for i in range(100):
            # 100 DISTINCT SMILES — a cycling pattern would collapse under
            # dedup and quietly make the cap assertion below untestable.
            fh.write("%.3f,%sO,%d_%d\n" % (i / 10.0, "C" * (i + 1), 1000 + i, 2000 + i))
        # a duplicate SMILES with a WORSE score, and a re-visit with a better one
        fh.write("0.5,CC,9001_9002\n")
        fh.write("99.0,CC,9003_9004\n")
        fh.write("notanumber,CCC,1_2\n")

    rows = R.read_results_csv(csv_path)
    check("unparseable row counted", R.read_results_csv.last_unparseable == 1,
          "got %d" % R.read_results_csv.last_unparseable)

    picked, st = R.select_top_fraction(rows, 0.30, None)
    check("dedup keeps the BEST occurrence, not the first",
          any(p["smiles"] == "CC" and p["score"] == 99.0 for p in picked))
    check("n_unique excludes the duplicate", st["n_unique"] == len(set(r["smiles"] for r in rows)),
          "n_unique=%d" % st["n_unique"])
    check("30%% of %d unique == %d" % (st["n_unique"], st["n_fraction"]),
          st["n_fraction"] == max(1, round(st["n_unique"] * 0.30)))
    check("sorted descending", all(picked[i]["score"] >= picked[i + 1]["score"]
                                   for i in range(len(picked) - 1)))

    picked2, st2 = R.select_top_fraction(rows, 0.30, 5)
    check("cap bites and says so", st2["capped"] is True and st2["n_selected"] == 5)
    check("cap does not rewrite n_fraction", st2["n_fraction"] == st["n_fraction"],
          "an honest report must still name the full selection size")

# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] credit assignment — a product scores BOTH its reagents")

records = [
    {"name": "111_222", "smiles": "C", "ok": True, "pk": 8.0},
    {"name": "111_333", "smiles": "CC", "ok": True, "pk": 6.0},
    {"name": "444_222", "smiles": "CCC", "ok": True, "dg": -6.8},     # pk = 5.0
    {"name": "555_666", "smiles": "CCCC", "ok": False, "err": "GIGN not configured"},
    {"name": "notaproduct", "smiles": "CCCCC", "ok": True, "pk": 9.9},
]
ev, evs = R.reagent_evidence(records)
check("failed record excluded", evs["n_no_score"] == 1)
check("unsplittable name excluded", evs["n_unresolved_name"] == 1)
check("3 usable records", evs["n_used"] == 3)
check("reagent 111 averages its two products", abs(ev["111"]["pk"] - 7.0) < 1e-9,
      "got %r" % ev.get("111"))
check("reagent 222 averages 8.0 and 5.0", abs(ev["222"]["pk"] - 6.5) < 1e-9,
      "got %r" % ev.get("222"))
check("dG converted at -dG/1.36", abs(ev["444"]["pk"] - 5.0) < 1e-9,
      "got %r" % ev.get("444"))
check("component index comes from position in the name",
      ev["111"]["comp"] == 0 and ev["222"]["comp"] == 1)

# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] round trip — merged checkpoint is loader-consumable, and the "
      "unmeasured survive")

with tempfile.TemporaryDirectory() as td:
    wdir = os.path.join(td, "Warmup_TS")
    os.makedirs(wdir)
    prev = {
        "rxn_key": "rxn110_suzuki", "timestamp": "20260810_100000",
        "prior_mean": 5.0, "prior_std": 1.0, "known_var": 1.0,
        "components": {
            "0": [{"reagent_name": "111", "current_mean": 5.4, "current_std": 0.40,
                   "known_var": 1.0, "num_scores": 3},
                  {"reagent_name": "444", "current_mean": 4.8, "current_std": 0.50,
                   "known_var": 1.0, "num_scores": 2},
                  {"reagent_name": "777", "current_mean": 6.1, "current_std": 0.30,
                   "known_var": 1.0, "num_scores": 9}],
            "1": [{"reagent_name": "222", "current_mean": 5.2, "current_std": 0.45,
                   "known_var": 1.0, "num_scores": 4},
                  {"reagent_name": "333", "current_mean": 4.9, "current_std": 0.60,
                   "known_var": 1.0, "num_scores": 1},
                  {"reagent_name": "888", "current_mean": 5.9, "current_std": 0.25,
                   "known_var": 1.0, "num_scores": 12}],
        },
        "n_components": 2, "n_reagents": 6,
    }
    with open(os.path.join(wdir, "suzuki_20260810_100000_warmup.json"), "w") as fh:
        json.dump(prev, fh)

    aff = {"schema": R.SCHEMA, "rxn_key": "rxn110_suzuki", "short_name": "suzuki",
           "scorer": "yupu_gign", "records": records}
    aff_path = os.path.join(td, "aff.json")
    with open(aff_path, "w") as fh:
        json.dump(aff, fh)

    rep = R.bias_from_affinity(aff_path, wdir, mapping="zblend", strength=0.5)
    check("bias_from_affinity ok", rep["ok"], rep.get("err", ""))
    ck_path = rep["checkpoint"]
    check("new checkpoint sorts above the old",
          os.path.basename(ck_path) > "suzuki_20260810_100000_warmup.json",
          os.path.basename(ck_path))
    check("latest_checkpoint now returns the new one",
          R.latest_checkpoint(wdir, "suzuki") == ck_path)

    with open(ck_path) as fh:
        ck = json.load(fh)

    # -- replay the loader's EXACT field access (ts_routes.py ~610-620) -------
    _prior_mean = ck["prior_mean"]
    _prior_std = ck["prior_std"]
    _known_var = _prior_std ** 2
    _belief_by_name = {}
    for _lst in ck.get("components", {}).values():
        for _r in _lst:
            _belief_by_name[_r["reagent_name"]] = _r

    restored = {}
    for rname in ["111", "222", "333", "444", "777", "888"]:
        b = _belief_by_name.get(rname)
        restored[rname] = {
            "mean": b["current_mean"] if b else _prior_mean,
            "std": b["current_std"] if b else _prior_std,
            "kv": (b.get("known_var") or _known_var) if b else _known_var,
            "n": b["num_scores"] if b else 0,
            "hit": bool(b),
        }
    check("every reagent is restorable by the loader",
          all(v["hit"] for v in restored.values()),
          [k for k, v in restored.items() if not v["hit"]])

    # THE trap: unmeasured reagents must come through byte-identical.
    check("unmeasured 777 untouched",
          restored["777"]["mean"] == 6.1 and restored["777"]["std"] == 0.30
          and restored["777"]["n"] == 9, restored["777"])
    check("unmeasured 888 untouched",
          restored["888"]["mean"] == 5.9 and restored["888"]["std"] == 0.25
          and restored["888"]["n"] == 12, restored["888"])
    check("carried count == the 2 unmeasured", rep["merge"]["n_carried_unchanged"] == 2,
          rep["merge"]["n_carried_unchanged"])

    # measured ones moved, in the right direction, by a bounded amount
    check("best reagent (111, pK 7.0 > batch mean) moved UP",
          restored["111"]["mean"] > 5.4, restored["111"]["mean"])
    check("worst reagent (444, pK 5.0 < batch mean) moved DOWN",
          restored["444"]["mean"] < 4.8, restored["444"]["mean"])
    biggest = max(abs(restored[r]["mean"] - prev["components"]["0" if r in ("111", "444") else "1"]
                      [[x["reagent_name"] for x in prev["components"]["0" if r in ("111", "444") else "1"]].index(r)]
                      ["current_mean"]) for r in ("111", "444", "222", "333"))
    check("no shift exceeds strength x prior_std x max|z|",
          biggest <= 0.5 * 1.0 * 2.0 + 1e-9, "biggest shift %.4f" % biggest)

    # std tightened, but neither collapsed nor RAISED
    floor = R.DEFAULT_STD_FLOOR * prev["prior_std"]
    check("measured std tightened", restored["111"]["std"] < 0.40,
          restored["111"]["std"])
    check("measured std respects the floor",
          all(restored[r]["std"] >= floor - 1e-9 for r in ("111", "222", "333", "444")),
          {r: restored[r]["std"] for r in ("111", "222", "333", "444")})
    before = {"111": 0.40, "444": 0.50, "222": 0.45, "333": 0.60}
    check("no measured std was RAISED by the clamp",
          all(restored[r]["std"] <= before[r] + 1e-9 for r in before),
          {r: (before[r], restored[r]["std"]) for r in before})

    # The regression this clamp exists for: a reagent already tighter than the
    # floor must come out unchanged, not loosened back up to it.
    conv = {"111": {"pk": 7.0, "n": 1, "comp": 0}}
    tight = {"prior_mean": 5.0, "prior_std": 1.0, "known_var": 1.0,
             "components": {"0": [{"reagent_name": "111", "current_mean": 5.4,
                                   "current_std": 0.01, "known_var": 1.0,
                                   "num_scores": 400}]}}
    nc, _ = R.merge_into_checkpoint(tight, conv, {"111": 0.1}, strength=0.5,
                                    std_floor=0.5)   # floor 0.5 >> old std 0.01
    got = nc["components"]["0"][0]["current_std"]
    check("a std already below an over-large floor is NOT raised", got <= 0.01 + 1e-9,
          "0.01 -> %s (the max()-only form would give 0.5)" % got)
    check("num_scores incremented by the measurement count",
          restored["111"]["n"] == 3 + 2 and restored["222"]["n"] == 4 + 2,
          (restored["111"]["n"], restored["222"]["n"]))

    # -- drift guard: the loader in ts_routes still reads these exact fields --
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "uiapp", "routes", "ts_routes.py")).read()
    for frag in ['belief["current_mean"]', 'belief["current_std"]',
                 'belief.get("known_var")', 'belief["num_scores"]',
                 '_ckpt["prior_mean"]', '_ckpt["prior_std"]']:
        check("loader still reads %s" % frag, frag in src,
              "the generated loader's schema changed — rl_bias must follow")
    check("loader still keys beliefs by reagent_name",
          re.search(r'_belief_by_name\[[^\]]*\["reagent_name"\]\]', src) is not None
          or '_r["reagent_name"]' in src)

    # -- mapping variants do not crash and stay bounded ----------------------
    for mp in ("rank", "raw"):
        r2 = R.bias_from_affinity(aff_path, wdir, mapping=mp, strength=0.5, dry_run=True)
        check("mapping=%s runs" % mp, r2["ok"], r2.get("err", ""))
        if mp == "raw":
            check("mapping=raw warns about the scale mismatch",
                  any("scale" in w for w in r2["mapping"]["warnings"]))

    # -- a critic that cannot separate the batch must move nothing -----------
    flat = {"schema": R.SCHEMA, "short_name": "suzuki", "records": [
        {"name": "111_222", "ok": True, "pk": 7.0},
        {"name": "444_333", "ok": True, "pk": 7.0}]}
    fp = os.path.join(td, "flat.json")
    with open(fp, "w") as fh:
        json.dump(flat, fh)
    r3 = R.bias_from_affinity(fp, wdir, mapping="zblend", dry_run=True)
    check("zero-variance batch moves no mean",
          all(a["mean_before"] == a["mean_after"] for a in r3["merge"]["applied"]),
          r3["merge"]["applied"])
    check("and says why", any("identical" in w for w in r3["mapping"]["warnings"]))

# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] no previous checkpoint is REFUSED (the destructive-reset trap)")

with tempfile.TemporaryDirectory() as td:
    wdir = os.path.join(td, "Warmup_TS")
    os.makedirs(wdir)
    aff_path = os.path.join(td, "aff.json")
    with open(aff_path, "w") as fh:
        json.dump({"schema": R.SCHEMA, "short_name": "suzuki", "records": records}, fh)
    try:
        R.bias_from_affinity(aff_path, wdir)
        check("refuses without a prior", False, "it wrote one anyway")
    except ValueError as e:
        check("refuses without a prior", "discarding" in str(e) or "prior" in str(e))
    r = R.bias_from_affinity(aff_path, wdir, allow_no_prior=True)
    check("allow_no_prior=True is an explicit escape hatch", r["ok"])

# ─────────────────────────────────────────────────────────────────────────────
print("\n[5] schema guard")
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "bad.json")
    with open(p, "w") as fh:
        json.dump({"schema": "something/else", "records": records}, fh)
    try:
        R.bias_from_affinity(p, td)
        check("wrong schema rejected", False)
    except ValueError:
        check("wrong schema rejected", True)

print("\n%s  %d failed" % ("FAILURES" if FAILED else "all passed", len(FAILED)))
if FAILED:
    for f in FAILED:
        print("   - " + f)
raise SystemExit(1 if FAILED else 0)
