#!/usr/bin/env python3
"""Pesi del modello fuori da git: push/pull verso un repo del Hugging Face Hub.

Perche': outputs/ e models/ (checkpoint, adapter, un tokenizer.json da 32 MB copiato 26 volte)
pesavano ~1 GB nel working tree e vivevano nella storia di git. Ora git li ignora e i pesi stanno
sul Hub, negli *stessi percorsi relativi* che gli script gia' usano ("outputs/gemma3-comprehensive",
"models/production/gemma3-ner-best", ...): dopo un pull, gli script esistenti funzionano senza modifiche.

    export GEMMA_NER_WEIGHTS_REPO=utente/gemma3-ner-weights    # oppure --repo
    export HF_TOKEN=...                                        # solo per push e per repo privati
    python scripts/weights_hub.py push outputs/gemma3-robust-training
    python scripts/weights_hub.py pull outputs/gemma3-robust-training
    python scripts/weights_hub.py pull --all
    python scripts/weights_hub.py list

Dal codice, scaricando solo se manca in locale:

    from scripts.weights_hub import ensure
    adapter_dir = ensure("outputs/gemma3-robust-training")

Note:
  * il token non si passa mai da riga di comando: HF_TOKEN o `huggingface-cli login`.
  * `push` crea il repo *privato* se non esiste (--public per cambiarlo); non modifica la
    visibilita' di un repo gia' esistente.
  * `push` esclude lo stato di ripresa dell'addestramento (optimizer.pt, scheduler.pt, rng_state*.pth);
    --with-training-state per includerlo.
"""
import argparse
import os
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TOP_LEVEL = ("outputs", "models")
ENV_REPO = "GEMMA_NER_WEIGHTS_REPO"
TRAINING_STATE = ["*optimizer.pt", "*scheduler.pt", "*rng_state*.pth"]


def check_name(name):
    """Accetta solo percorsi relativi sotto outputs/ o models/ (niente '..', niente assoluti)."""
    p = PurePosixPath(str(name).replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts or len(p.parts) < 2 or p.parts[0] not in TOP_LEVEL:
        raise ValueError(f"nome non valido: {name!r} (atteso: outputs/<nome> o models/<nome>)")
    return p.as_posix()


def resolve_repo(repo=None):
    repo = repo or os.environ.get(ENV_REPO)
    if not repo:
        raise SystemExit(f"manca il repo del Hub: usa --repo utente/nome oppure imposta {ENV_REPO}")
    return repo


def _present(name):
    path = ROOT / name
    return path.is_dir() and any(path.iterdir())


def pull(names=None, repo=None, everything=False):
    from huggingface_hub import snapshot_download

    repo = resolve_repo(repo)
    names = [check_name(n) for n in (names or [])]
    if not names and not everything:
        raise SystemExit("indica cosa scaricare (NOME ...) oppure --all")
    patterns = [f"{n}/*" for n in names] if names else [f"{t}/*" for t in TOP_LEVEL]
    snapshot_download(repo_id=repo, repo_type="model", local_dir=str(ROOT), allow_patterns=patterns)
    missing = [n for n in names if not _present(n)]
    if missing:
        raise FileNotFoundError(f"non trovato su {repo}: {', '.join(missing)}")
    return [ROOT / n for n in names]


def push(names, repo=None, private=True, with_training_state=False):
    from huggingface_hub import HfApi

    repo = resolve_repo(repo)
    names = [check_name(n) for n in names]
    absent = [n for n in names if not _present(n)]
    if absent:
        raise SystemExit(f"non esiste in locale (o e' vuoto): {', '.join(absent)}")
    api = HfApi()
    api.create_repo(repo_id=repo, repo_type="model", private=private, exist_ok=True)
    ignore = ["__pycache__/*", "*.pyc"] + ([] if with_training_state else TRAINING_STATE)
    for n in names:
        api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(ROOT / n),
                          path_in_repo=n, ignore_patterns=ignore, commit_message=f"Add {n}")
        print(f"caricato: {n} -> {repo}")


def ensure(name, repo=None):
    """Ritorna il percorso locale di `name`, scaricandolo dal Hub solo se manca."""
    name = check_name(name)
    if not _present(name):
        pull([name], repo=repo)
    return str(ROOT / name)


def list_remote(repo=None):
    from huggingface_hub import HfApi

    repo = resolve_repo(repo)
    groups = {}
    for f in HfApi().list_repo_files(repo_id=repo, repo_type="model"):
        parts = PurePosixPath(f).parts
        key = "/".join(parts[:3]) if parts[0] == "models" and len(parts) > 3 else "/".join(parts[:2])
        groups[key] = groups.get(key, 0) + 1
    for key in sorted(groups):
        print(f"{groups[key]:5d} file  {key}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", help=f"repo del Hub (default: ${ENV_REPO})")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_pull = sub.add_parser("pull", help="scarica pesi dal Hub negli stessi percorsi")
    p_pull.add_argument("names", nargs="*", metavar="NOME")
    p_pull.add_argument("--all", action="store_true", help="tutto outputs/ e models/")
    p_push = sub.add_parser("push", help="carica cartelle locali sul Hub")
    p_push.add_argument("names", nargs="+", metavar="NOME")
    p_push.add_argument("--public", action="store_true", help="crea il repo pubblico (default: privato)")
    p_push.add_argument("--with-training-state", action="store_true", help="includi optimizer/scheduler/rng")
    sub.add_parser("list", help="elenca cosa c'e' sul Hub")
    args = ap.parse_args(argv)
    if args.cmd == "pull":
        pull(args.names, repo=args.repo, everything=args.all)
    elif args.cmd == "push":
        push(args.names, repo=args.repo, private=not args.public,
             with_training_state=args.with_training_state)
    else:
        list_remote(repo=args.repo)


if __name__ == "__main__":
    sys.exit(main())
