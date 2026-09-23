#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""compare_arch.py -- the checkpoint's architecture vs the one shufflenet_v3.py builds.

    python compare_arch.py \
        --model    /home/huangzihang/repos/user_data/deepatom/checkpoints/best_val_rmse.pth.tar \
        --repo-dir /home/huangzihang/repos/elion/src/elion/properties/deepatom/model_split_data/02_pytorch

Prints, for each side: the stage/block structure, the channel widths, the
parameter count, and the full name->shape tree (collapsed per block). Then the
exact diff: keys only in the checkpoint, keys only in the model, and every shape
mismatch.

It also hunts for where `repeat_block` actually comes from -- the constructor,
a module constant, an imported config module -- because that is what decides
whether the two can be reconciled without swapping the file.

    --tree       print every parameter, not one line per block
    --width 2    only try that width (default: 2, 1, 0.5)
"""
from __future__ import annotations

import argparse
import re
import sys
import types
from pathlib import Path

_INIT_PARAMS_SRC = """
import torch.nn as nn
import torch.nn.init as init


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv3d):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)
"""

WIDTHS = {"ShuffleNetV3x2": 2.0, "ShuffleNetV3x1": 1.0, "ShuffleNetV3x0.5": 0.5}


def load_module(repo_dir):
    import importlib
    repo_dir = str(Path(repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    if "utilies" not in sys.modules:
        stub = types.ModuleType("utilies")
        exec(compile(_INIT_PARAMS_SRC, "<stub>", "exec"), stub.__dict__)
        sys.modules["utilies"] = stub
    mod = importlib.import_module("shufflenet_v3")
    return mod, mod.ShuffleNetV3


def depths(sd):
    out = {}
    for k in sd:
        m = re.match(r"^(stage\d+)\.(\d+)\.", k)
        if m:
            out[m.group(1)] = max(out.get(m.group(1), -1), int(m.group(2)))
    return {k: v + 1 for k, v in sorted(out.items())}


def summarise(tag, sd):
    n = sum(int(v.numel()) for v in sd.values() if hasattr(v, "numel"))
    d = depths(sd)
    print("%s" % tag)
    print("  tensors      %d" % len(sd))
    print("  parameters   %s" % format(n, ","))
    print("  stages       %s" % (", ".join("%s=%d" % kv for kv in d.items()) or "(none named stageN)"))
    # channel widths: the out_channels of the first conv in each stage
    chans = {}
    for k, v in sd.items():
        m = re.match(r"^(stage\d+)\.0\..*weight$", k)
        if m and hasattr(v, "shape") and len(v.shape) == 5 and m.group(1) not in chans:
            chans[m.group(1)] = int(v.shape[0])
    if chans:
        print("  stage width  %s" % ", ".join("%s->%d" % kv for kv in sorted(chans.items())))
    first = [k for k in sd if k.startswith("conv1.")]
    if first:
        s = sd[first[0]].shape
        print("  input conv   %s  (in_channels=%d, out=%d)" % (tuple(s), s[1], s[0]))
    return d


def tree(sd, full=False, limit=400):
    """name -> shape, collapsed to one line per stage.block unless --tree."""
    rows = [(k, tuple(v.shape)) for k, v in sd.items() if hasattr(v, "shape")]
    if full:
        for k, s in rows:
            print("    %-58s %s" % (k, s))
        return
    seen = {}
    for k, s in rows:
        m = re.match(r"^(stage\d+\.\d+)\.", k)
        key = m.group(1) if m else k.split(".")[0]
        seen.setdefault(key, []).append((k, s))
    for key, items in list(seen.items())[:limit]:
        if len(items) == 1:
            print("    %-58s %s" % (items[0][0], items[0][1]))
        else:
            shapes = ", ".join(str(s) for _, s in items[:3])
            more = "  (+%d more)" % (len(items) - 3) if len(items) > 3 else ""
            print("    %-58s %d tensors: %s%s" % (key + ".*", len(items), shapes, more))


def find_repeat_block(mod, repo_dir):
    """Where could the [3,4,4] be coming from? Report every candidate."""
    print("\nwhere repeat_block lives")
    hits = []
    repo = str(Path(repo_dir).resolve())

    import inspect
    try:
        sig = inspect.signature(mod.ShuffleNetV3.__init__)
        print("  constructor  ShuffleNetV3%s" % str(sig).replace("(self, ", "("))
        if "repeat_block" in sig.parameters:
            hits.append("constructor keyword repeat_block")
    except (TypeError, ValueError):
        pass

    def scan(m, label):
        for attr in dir(m):
            if attr.startswith("__"):
                continue
            try:
                val = getattr(m, attr)
            except Exception:
                continue
            if isinstance(val, (list, tuple)) and 2 <= len(val) <= 5 \
                    and all(isinstance(v, int) for v in val):
                print("  %-26s %s = %s" % (label, attr, list(val)))
                hits.append("%s.%s" % (label, attr))

    scan(mod, "shufflenet_v3")
    for name, m in list(sys.modules.items()):
        if m is None or m is mod or not hasattr(m, "__file__") or not m.__file__:
            continue
        try:
            if str(Path(m.__file__).resolve()).startswith(repo):
                scan(m, name)
        except OSError:
            continue

    # class-level attributes too
    for attr in dir(mod.ShuffleNetV3):
        if attr.startswith("__"):
            continue
        val = getattr(mod.ShuffleNetV3, attr, None)
        if isinstance(val, (list, tuple)) and 2 <= len(val) <= 5 \
                and all(isinstance(v, int) for v in val):
            print("  %-26s %s = %s" % ("ShuffleNetV3 (class)", attr, list(val)))
            hits.append("ShuffleNetV3.%s" % attr)

    src = None
    try:
        src = Path(mod.__file__).read_text(errors="replace")
    except Exception:
        pass
    if src:
        for m in re.finditer(r"^.*repeat_block.*$", src, re.M):
            line = m.group(0).strip()
            if len(line) < 200:
                print("  source line                %s" % line)
    if not hits:
        print("  -- nothing found; it is computed inside __init__ or imported "
              "from outside the repo dir")
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument("--tree", action="store_true")
    a = ap.parse_args()

    import torch
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}

    print("=" * 78)
    ck_depth = summarise("CHECKPOINT   %s" % a.model, sd)
    if isinstance(ck, dict):
        print("  epoch        %s      val_rmse %s" % (ck.get("epoch"), ck.get("val_rmse")))
    print("  ---- structure ----")
    tree(sd, a.tree)

    mod, Net = load_module(a.repo_dir)
    print("\n" + "=" * 78)
    print("SHUFFLENET   %s" % mod.__file__)

    widths = [a.width] if a.width else [2.0, 1.0, 0.5]
    best = None
    for w in widths:
        try:
            model = Net(input_channel=int(sd["conv1.weight"].shape[1])
                        if "conv1.weight" in sd else 24,
                        dropout_prob=0.0, width_multiplier=w)
        except Exception as e:
            print("\n  width %.1f -- could not construct: %s" % (w, e))
            continue
        msd = model.state_dict()
        missing = [k for k in msd if k not in sd]
        unexpected = [k for k in sd if k not in msd]
        badshape = [(k, tuple(sd[k].shape), tuple(msd[k].shape))
                    for k in msd if k in sd and tuple(msd[k].shape) != tuple(sd[k].shape)]
        score = len(missing) + len(unexpected) + len(badshape)
        if best is None or score < best[0]:
            best = (score, w, msd, missing, unexpected, badshape)

    if best is None:
        sys.exit("could not construct the model at any width")
    score, w, msd, missing, unexpected, badshape = best
    print("\n  closest width %.1f" % w)
    md_depth = summarise("  MODEL as built", msd)
    print("  ---- structure ----")
    tree(msd, a.tree)

    print("\n" + "=" * 78)
    print("DIFF   (checkpoint vs model)")
    print("  stages   checkpoint %s   model %s"
          % (list(ck_depth.values()), list(md_depth.values())))
    print("\n  in the MODEL but not the checkpoint  (%d)" % len(missing))
    for k in missing[:25]:
        print("    %-58s %s" % (k, tuple(msd[k].shape)))
    if len(missing) > 25:
        print("    ... %d more" % (len(missing) - 25))
    print("\n  in the CHECKPOINT but not the model  (%d)" % len(unexpected))
    for k in unexpected[:25]:
        print("    %-58s %s" % (k, tuple(sd[k].shape)))
    if len(unexpected) > 25:
        print("    ... %d more" % (len(unexpected) - 25))
    print("\n  SHAPE mismatches  (%d)" % len(badshape))
    for k, cs, ms in badshape[:25]:
        print("    %-46s ckpt %-18s model %s" % (k, cs, ms))
    if len(badshape) > 25:
        print("    ... %d more" % (len(badshape) - 25))

    find_repeat_block(mod, a.repo_dir)

    print("\n" + "=" * 78)
    if score == 0:
        print("IDENTICAL -- this checkpoint fits this shufflenet_v3.py.")
    elif not unexpected and ck_depth != md_depth:
        print("Same family, different DEPTH. Every checkpoint tensor has a home in")
        print("the model; the model simply has extra blocks. Reconcilable only by")
        print("building at %s -- see 'where repeat_block lives' above for the knob,"
              % list(ck_depth.values()))
        print("or use the shufflenet_v3.py the checkpoint was trained against.")
    else:
        print("These are different networks.")


if __name__ == "__main__":
    sys.exit(main())
