#!/usr/bin/env python3
"""Check train/validation/test leakage in the NER datasets.

Every ``*.jsonl`` under the given roots whose rows have ``document`` and ``output`` is classified
by file name: ``train`` -> training, ``val`` / ``test`` -> evaluation. The script reports, for each
evaluation file:

  * documents that also appear (after normalisation) in a training file;
  * near-duplicates: word-set Jaccard similarity against any training document;
  * entities (same type, same string) that already occur in a training output;
  * documents shared between evaluation files.

Run from the repository root:

    python scripts/check_overlap.py                      # report only
    python scripts/check_overlap.py --fail-on-leak       # exit 1 on any exact overlap
    python scripts/check_overlap.py --write-clean OUT    # also write de-duplicated training files

``--write-clean`` never touches the originals: it writes copies of the training files into OUT,
dropping rows whose document is an exact duplicate of, or at least ``--clean-threshold`` similar to,
an evaluation document. Re-train and re-evaluate on those copies before quoting any metric.

Jaccard on word sets is a heuristic: templated documents share many words, so read the near-duplicate
counts as an upper bound on contamination. Exact duplicates are unambiguous.
"""
import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_ROOTS = ["data/ner", "data/expanded"]


def normalise(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def load(path):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def role(path):
    name = path.name.lower()
    if "train" in name:
        return "train"
    if "val" in name or "test" in name:
        return "eval"
    return None


def entities(row):
    out = row.get("output")
    try:
        out = json.loads(out) if isinstance(out, str) else out
    except ValueError:
        return set()
    found = set()
    if isinstance(out, dict):
        for kind, values in out.items():
            for value in values if isinstance(values, list) else [values]:
                if isinstance(value, str):
                    found.add((kind, normalise(value)))
    return found


def discover(roots):
    files = {}
    for root in roots:
        for path in sorted(Path(root).glob("*.jsonl")):
            if role(path) is None:
                continue
            rows = load(path)
            if rows and "document" in rows[0] and "output" in rows[0]:
                files[path] = rows
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", help="directory with *.jsonl files (repeatable)")
    ap.add_argument("--near", type=float, default=0.7, help="Jaccard threshold reported as near-duplicate")
    ap.add_argument("--clean-threshold", type=float, default=0.9, help="Jaccard threshold used by --write-clean")
    ap.add_argument("--fail-on-leak", action="store_true", help="exit 1 if an evaluation document is also in a training file")
    ap.add_argument("--write-clean", metavar="DIR", help="write de-duplicated copies of the training files into DIR")
    args = ap.parse_args()

    files = discover(args.root or DEFAULT_ROOTS)
    train = {p: r for p, r in files.items() if role(p) == "train"}
    evals = {p: r for p, r in files.items() if role(p) == "eval"}
    if not train or not evals:
        sys.exit("no training/evaluation files found (run from the repository root, or pass --root)")

    docs = {p: [normalise(r["document"]) for r in rows] for p, rows in files.items()}
    words = {p: [set(d.split()) for d in ds] for p, ds in docs.items()}
    train_docs = {d for p in train for d in docs[p]}
    train_words = [w for p in train for w in words[p]]
    train_entities = set().union(*(entities(r) for rows in train.values() for r in rows))

    leaked = 0
    print(f"{'evaluation file':44s} {'docs':>5s} {'exact':>6s} {'>=0.9':>6s} {'>=' + str(args.near):>6s} {'entities seen':>15s}")
    for path, rows in evals.items():
        best = [max((jaccard(w, t) for t in train_words), default=0.0) for w in words[path]]
        exact = sum(d in train_docs for d in docs[path])
        total = seen = 0
        for row in rows:
            ents = entities(row)
            total += len(ents)
            seen += len(ents & train_entities)
        leaked += exact
        print(f"{str(path):44s} {len(rows):5d} {exact:6d} {sum(b >= 0.9 for b in best):6d} "
              f"{sum(b >= args.near for b in best):6d} {seen:6d}/{total:<4d} {100 * seen / max(1, total):3.0f}%")

    print("\nDocuments shared between evaluation files:")
    paths = list(evals)
    shared_any = False
    for i, a in enumerate(paths):
        for b in paths[i + 1:]:
            n = len(set(docs[a]) & set(docs[b]))
            if n:
                shared_any = True
                print(f"  {a} and {b}: {n}")
    if not shared_any:
        print("  none")

    if args.write_clean:
        out_dir = Path(args.write_clean)
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_docs = {d for p in evals for d in docs[p]}
        eval_words = [w for p in evals for w in words[p]]
        print(f"\nWriting de-duplicated training files to {out_dir}/ (threshold {args.clean_threshold})")
        for path, rows in train.items():
            keep = [
                row for row, d, w in zip(rows, docs[path], words[path])
                if d not in eval_docs and all(jaccard(w, e) < args.clean_threshold for e in eval_words)
            ]
            target = out_dir / f"{path.parent.name}__{path.name}"
            with open(target, "w", encoding="utf-8") as fh:
                for row in keep:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  {str(path):44s} {len(rows):5d} -> {len(keep):5d}  ({len(rows) - len(keep)} dropped)  {target}")

    if args.fail_on_leak and leaked:
        sys.exit(f"\nLEAK: {leaked} evaluation documents also appear in a training file")


if __name__ == "__main__":
    main()
