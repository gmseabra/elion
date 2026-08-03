#!/usr/bin/env python
"""
Building-block database scanner.

Given a directory of reagent CSVs, reports for every reaction in the catalogue
how many building blocks in each file are eligible as the FIRST reagent and how
many as the SECOND — i.e. how many match each reactant template of the
reaction's SMARTS.

    python <repo>/uiapp/core/bb_scan.py --path <dir> --catalogue <catalogue.json>

Invoke it by PATH, never with ``-m uiapp.core.bb_scan``: the ``-m`` form imports
the ``uiapp`` package first, which imports the whole route layer and therefore
torch — defeating the isolation this file exists for, and failing outright on an
interpreter that has RDKit but not torch. Nothing here imports ``uiapp``, so run
by path it stands alone.

Emits one JSON object per line on stdout (start / file / done / error), so the
caller can stream progress. Written as a standalone script rather than an
in-process helper for two reasons, both load-bearing:

  1. **RDKit can segfault.** A substructure match is C++; a crash takes the
     whole interpreter with it. In-process that would kill Flask mid-scan and
     with it every running Thompson-Sampling job — a multi-hour run destroyed by
     someone clicking a button on a bad CSV. Out of process, a crash is an exit
     code the route reports.
  2. The scan is CPU-bound for minutes on a large library. A subprocess keeps it
     off the request thread and makes it killable.

RDKIT FOOTGUN THIS AVOIDS
-------------------------
The obvious way to get the reactant patterns is::

    rxn = AllChem.ReactionFromSmarts(smarts)
    query = rxn.GetReactantTemplate(0)          # borrowed reference!

`GetReactantTemplate` returns a reference *into* the ChemicalReaction. If the
reaction object is garbage-collected — including the very common
``AllChem.ReactionFromSmarts(s).GetReactantTemplate(0)`` one-liner, or a loop
that rebinds ``rxn`` while keeping the templates — the query dangles and the
next `HasSubstructMatch` segfaults. Verified: it crashes on the first molecule.

So the queries here are built with `Chem.MolFromSmarts()` on the reactant side,
split at top-level '.' — those are independent, owned objects. The reaction is
still parsed once, purely to validate the SMARTS and to check that the number of
components agrees with the number of reactant templates.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time

# A reagent file is a molecule list, not a data set; these bounds exist so a
# mistyped path (say, / ) reports a bounded, honest answer instead of running
# for hours. Both are reported in the output when they bite.
DEFAULT_MAX_ROWS = 200_000
DEFAULT_MAX_FILES = 500
# Duplicate detection keeps one 64-bit hash per distinct molecule. At ~32 bytes
# per Python int plus set overhead that is roughly 240 MB for 3.2M uniques, so
# it is capped rather than allowed to grow until the machine swaps. Past the cap
# the scan continues and the report says the count is a lower bound — a wrong
# number presented as exact would be worse than an honest partial one.
DEFAULT_MAX_UNIQUE = 5_000_000
# Per file, how many duplicate examples to carry back for the UI. The counts are
# exact; these are illustrative rows behind a click, and a file with 30k
# duplicates must not put 30k records on the wire.
DEFAULT_MAX_DETAILS = 25
# Duplicate records written to disk per file, for the paged viewer. The inline
# cap above is what rides the SSE stream; this is what the pager reads back.
# ~200 B a record, so 5000 is ~1 MB per file and 1000 pages at 5 rows — far past
# what anyone pages through, while keeping a 200-file scan under ~200 MB.
DEFAULT_MAX_RECORDS = 5_000
SMILES_HEADER_HINTS = ("smiles", "smi", "structure", "canonical_smiles")


def emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def split_top_level(smarts: str) -> list[str]:
    """Split a SMARTS reactant side on '.' that are NOT inside [] or ().

    A plain ``smarts.split('.')`` is wrong: recursive SMARTS such as
    ``$(N-C=[O,N,S])`` and atom lists routinely contain characters that make a
    naive split produce unparseable fragments, and `MolFromSmarts` answers None
    for those — which would silently read as "zero eligible building blocks"
    rather than as an error.
    """
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in smarts:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "." and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [c for c in (s.strip() for s in out) if c]


def build_queries(catalogue: dict) -> tuple[dict, list[str]]:
    """{reaction_key: [query_mol, ...]} plus a list of human-readable problems."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    queries: dict = {}
    problems: list[str] = []
    for key, entry in catalogue.items():
        smarts = (entry or {}).get("smarts") or ""
        if not smarts:
            problems.append(f"{key}: no smarts in catalogue")
            continue
        comps = split_top_level(smarts.split(">>")[0])

        # Parse the reaction only to validate. Keep `rxn` alive for the whole
        # check — see the module docstring for why that matters.
        try:
            rxn = AllChem.ReactionFromSmarts(smarts)
            if rxn is not None and rxn.GetNumReactantTemplates() != len(comps):
                problems.append(
                    f"{key}: {len(comps)} component(s) split from the SMARTS but the "
                    f"reaction parser reports {rxn.GetNumReactantTemplates()} reactant "
                    f"template(s) — the split is wrong, counts would be meaningless")
                continue
            del rxn
        except Exception as exc:
            problems.append(f"{key}: SMARTS did not parse as a reaction ({exc})")
            continue

        mols = []
        for i, comp in enumerate(comps):
            q = Chem.MolFromSmarts(comp)
            if q is None:
                problems.append(f"{key}: reactant {i + 1} is not valid SMARTS: {comp}")
                break
            mols.append(q)
        else:
            queries[key] = mols
    return queries, problems


def read_rows(path: str, max_rows: int) -> tuple[list, int, bool]:
    """Return ([(smiles, bb_id, line_no), ...], total_rows_seen, truncated).

    `line_no` is the 1-based line in the file INCLUDING the header, so it is the
    number an editor shows — the report says "row 4213" and `sed -n 4213p` lands
    on it. The id column is optional; without one the row number is the identity.
    """
    rows: list = []
    total = 0
    truncated = False
    with open(path, newline="", errors="replace") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        reader = csv.reader(fh, dialect)
        try:
            header = next(reader)
        except StopIteration:
            return [], 0, False
        line_no = 1

        col = None
        for i, name in enumerate(header):
            if name.strip().lower().lstrip("#").strip() in SMILES_HEADER_HINTS:
                col = i
                break
        id_col = None
        for i, name in enumerate(header):
            n = name.strip().lower()
            if i != col and ("id" in n or n in ("catalog", "cat", "code")):
                id_col = i
                break

        if col is None:
            # No recognised header: row 1 is data, not names.
            col = 0
            if header and header[0].strip():
                if not any(h.strip().lower() in SMILES_HEADER_HINTS for h in header):
                    rows.append((header[0].strip(),
                                 header[1].strip() if len(header) > 1 else "", 1))
                    total += 1

        for row in reader:
            line_no += 1
            total += 1
            if len(rows) >= max_rows:
                truncated = True
                continue
            if col < len(row):
                val = row[col].strip()
                if val:
                    bb_id = row[id_col].strip() if (id_col is not None and id_col < len(row)) else ""
                    rows.append((val, bb_id, line_no))
    return rows, total, truncated


def read_smiles(path: str, max_rows: int) -> tuple[list[str], int, bool]:
    """SMILES only — thin wrapper over read_rows."""
    rows, total, truncated = read_rows(path, max_rows)
    return [r[0] for r in rows], total, truncated


class Dedup:
    """Molecule identity across the whole scan.

    ``seen`` maps a canonical hash to ``"<file_idx>|<row>|<id>"`` — the FIRST
    place that molecule was found — because "duplicate of what?" is unanswerable
    from a set of hashes. That provenance roughly doubles the memory (one short
    string per unique molecule, ~250 MB at 1.4M uniques) and is the price of the
    report naming the original instead of only counting it.
    """

    def __init__(self, max_unique: int = DEFAULT_MAX_UNIQUE,
                 max_details: int = DEFAULT_MAX_DETAILS,
                 details_dir: str = "", max_records: int = DEFAULT_MAX_RECORDS):
        self.seen: dict = {}
        self.files: list = []          # file_idx -> basename
        self.max_unique = max_unique
        self.max_details = max_details
        self.details_dir = details_dir
        self.max_records = max_records
        self.duplicates = 0
        self.capped = False

    def add_file(self, basename: str) -> int:
        self.files.append(basename)
        return len(self.files) - 1


def canonical(mol) -> str:
    """The canonical SMILES — the string that defines molecular identity here."""
    from rdkit import Chem
    return Chem.MolToSmiles(mol)


def hash_of(canonical_smiles: str) -> int:
    """64-bit hash of a canonical SMILES.

    Hashed rather than stored because the strings are the memory, not the
    comparison. Over 3.2M molecules the chance of a single collision is ~3e-7.
    """
    import hashlib
    return int.from_bytes(
        hashlib.blake2b(canonical_smiles.encode(), digest_size=8).digest(), "big")


def canonical_hash(mol) -> int:
    """Identity hash for `mol`. Canonical, so the same molecule written two ways
    collapses to one — see the duplicate-detection notes in tests/test_bb_scan.py."""
    return hash_of(canonical(mol))


def scan_file(path: str, queries: dict, max_rows: int,
              dedup: "Dedup | None" = None) -> dict:
    from rdkit import Chem

    t0 = time.time()
    rows, total_rows, truncated = read_rows(path, max_rows)
    base = os.path.basename(path)
    file_idx = dedup.add_file(base) if dedup is not None else 0

    mols = []
    unparseable = 0
    duplicates = 0
    new_unique = 0
    dedup_capped = False
    details: list = []
    details_capped = False

    # Full duplicate records go to disk so the UI can page through a file with
    # 10k+ of them. The SSE stream carries only the first `max_details` — a
    # count that large cannot ride the wire, and holding it in the browser would
    # be worse than reading a slice back on demand.
    rec_fh = None
    rec_written = 0
    rec_truncated = False
    if dedup is not None and dedup.details_dir:
        try:
            os.makedirs(dedup.details_dir, exist_ok=True)
            rec_fh = open(os.path.join(dedup.details_dir, f"f{file_idx}.jsonl"), "w")
        except OSError:
            rec_fh = None

    for smi, bb_id, line_no in rows:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            unparseable += 1
            continue
        mols.append(m)
        if dedup is None:
            continue

        can = canonical(m)
        h = hash_of(can)
        prev = dedup.seen.get(h)
        if prev is not None:
            duplicates += 1
            dedup.duplicates += 1
            if rec_fh is not None or len(details) < dedup.max_details:
                p_idx, p_row, p_id = prev.split("|", 2)
                rec = {
                    # The canonical form is the identity; `raw` is how THIS file
                    # wrote it. When they differ, that difference is the finding:
                    # a text-level dedup would have missed the row entirely.
                    "smiles": can,
                    "raw":    smi if smi != can else "",
                    "id":     bb_id,
                    "row":    line_no,
                    "of_file": dedup.files[int(p_idx)] if int(p_idx) < len(dedup.files) else "?",
                    "of_row":  int(p_row),
                    "of_id":   p_id,
                }
                if len(details) < dedup.max_details:
                    details.append(rec)
                else:
                    details_capped = True
                if rec_fh is not None:
                    if rec_written < dedup.max_records:
                        rec_fh.write(json.dumps(rec) + "\n")
                        rec_written += 1
                    else:
                        rec_truncated = True
            else:
                details_capped = True
        else:
            if len(dedup.seen) >= dedup.max_unique:
                dedup_capped = True
                dedup.capped = True
                continue
            new_unique += 1
            dedup.seen[h] = f"{file_idx}|{line_no}|{bb_id}"

    if rec_fh is not None:
        rec_fh.close()

    counts: dict = {}
    for key, qs in queries.items():
        counts[key] = [sum(1 for m in mols if m.HasSubstructMatch(q)) for q in qs]

    return {
        "type":           "file",
        "idx":            file_idx,
        "dup_written":    rec_written,
        "dup_truncated":  rec_truncated,
        "file":           base,
        "path":           path,
        "rows":           total_rows,
        "parsed":         len(mols),
        "unparseable":    unparseable,
        "truncated":      truncated,
        "duplicates":     duplicates,
        "new_unique":     new_unique,
        "dedup_capped":   dedup_capped,
        "dup_details":    details,
        "details_capped": details_capped,
        "counts":         counts,
        "seconds":        round(time.time() - t0, 2),
    }


def find_csvs(root: str, max_files: int) -> tuple[list[str], bool]:
    found = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
    return found[:max_files], len(found) > max_files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True, help="directory of reagent CSVs")
    ap.add_argument("--catalogue", required=True, help="JSON file: {key: {smarts, short_name}}")
    ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    ap.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    ap.add_argument("--max-unique", type=int, default=DEFAULT_MAX_UNIQUE)
    ap.add_argument("--max-details", type=int, default=DEFAULT_MAX_DETAILS,
                    help="duplicate examples carried back per file on the stream")
    ap.add_argument("--details-dir", default="",
                    help="write full duplicate records here as f<idx>.jsonl "
                         "so the UI can page through them")
    ap.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS,
                    help="duplicate records written to disk per file")
    ap.add_argument("--no-dedup", action="store_true",
                    help="skip duplicate detection (saves canonicalisation time "
                         "and memory on very large libraries)")
    args = ap.parse_args(argv)

    t_start = time.time()
    root = os.path.expanduser(args.path.strip())

    if not os.path.exists(root):
        emit({"type": "error", "message": f"No such path: {root}"})
        return 2
    if not os.path.isdir(root):
        emit({"type": "error", "message": f"Not a directory: {root}"})
        return 2

    try:
        with open(args.catalogue) as fh:
            catalogue = json.load(fh)
    except Exception as exc:
        emit({"type": "error", "message": f"Cannot read catalogue: {exc}"})
        return 2

    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        emit({"type": "error",
              "message": "RDKit is not installed in this interpreter — the scan needs it "
                         "to match building blocks against the reaction SMARTS."})
        return 3

    queries, problems = build_queries(catalogue)
    for p in problems:
        emit({"type": "warning", "message": p})
    if not queries:
        emit({"type": "error", "message": "No usable reaction SMARTS in the catalogue."})
        return 4

    files, capped = find_csvs(root, args.max_files)
    if not files:
        emit({"type": "error", "message": f"No .csv files found under {root}"})
        return 5

    emit({"type": "start", "root": root, "n_files": len(files),
          "files_capped": capped, "max_files": args.max_files,
          "dedup": not args.no_dedup,
          "details_dir": args.details_dir,
          "reactions": {k: (catalogue.get(k) or {}).get("short_name", k) for k in queries},
          "n_reactants": {k: len(v) for k, v in queries.items()}})

    totals: dict = {k: [0] * len(v) for k, v in queries.items()}
    # One set for the whole scan: duplicates are counted ACROSS files, because
    # the question the report answers is "how many distinct building blocks does
    # this library actually contain", not "how many does each file repeat".
    dedup = (None if args.no_dedup
             else Dedup(args.max_unique, args.max_details,
                        args.details_dir, args.max_records))
    n_dupes = 0
    n_rows = 0
    dedup_capped = False

    for idx, path in enumerate(files):
        try:
            rec = scan_file(path, queries, args.max_rows, dedup)
        except Exception as exc:
            emit({"type": "file_error", "file": os.path.basename(path),
                  "path": path, "message": str(exc)})
            continue
        rec["index"] = idx
        rec["of"] = len(files)
        emit(rec)
        for key, cs in rec["counts"].items():
            for i, c in enumerate(cs):
                totals[key][i] += c
        n_dupes += rec.get("duplicates", 0)
        n_rows += rec.get("parsed", 0)
        dedup_capped = dedup_capped or rec.get("dedup_capped", False)

    emit({"type": "done", "totals": totals, "n_files": len(files),
          "dedup": dedup is not None,
          "parsed_total": n_rows,
          "duplicate_total": n_dupes,
          "unique_total": (len(dedup.seen) if dedup is not None else None),
          "dedup_capped": dedup_capped,
          "max_unique": args.max_unique,
          "seconds": round(time.time() - t_start, 2)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())