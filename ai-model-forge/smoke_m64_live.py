"""M64 LIVE certification smoke (READ-ONLY, executed exactly ONCE).

Certifies `GET /api/v1/datasets/{id}/usage` and
`GET /api/v1/tokenizers/{id}/usage` against the production root.
Every reference is cross-checked against an INDEPENDENT oracle that
parses the raw manifests/filesystem directly (no app code), and the
evaluation/comparison categories additionally against the public
cross-reference routes. Zero state-mutating calls: GET only.

Expected storage outcome: byte-for-byte identical before and after
(verified against the pre-captured m64_pre.sha256 AND a before/after
walk).

Certified facts of this production root (identical bytes since the M63
certification — m64_pre.sha256 == m63_pre.sha256, 52/52):
  1 dataset (ccf20044baff), 1 tokenizer (8a54196b1589), 1 model;
  dataset usage: training_run 6, workflow 2, evaluation 4,
  comparison 2, suite_run 0, tokenizer_training 1,
  tokenized_version 1 -> total 16;
  tokenizer usage: training_run 6, workflow 2, evaluation 4,
  comparison 2, suite_run 0, sample 0, sample_quality 0,
  tokenized_dataset 1 -> total 15.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8786"
API = BASE + "/api/v1"

# ---- certified constants of THIS root (facts, hard-coded on purpose) ----
DS_ID = "ccf20044baff"
TOK_ID = "8a54196b1589"
MODEL_ID = "70aa05e0bee6"
DS_NAME = "live-corpus"
TOK_NAME = "live-tok"
DS_TOTAL = 16
TOK_TOTAL = 15
DS_CATS = {"training_run": 6, "workflow": 2, "evaluation": 4,
           "comparison": 2, "suite_run": 0, "tokenizer_training": 1,
           "tokenized_version": 1}
TOK_CATS = {"training_run": 6, "workflow": 2, "evaluation": 4,
            "comparison": 2, "suite_run": 0, "sample": 0,
            "sample_quality": 0, "tokenized_dataset": 1}
N_CKPTS, N_DELETABLE, N_PROTECTED = 13, 10, 3
PRE_INVENTORY = Path(__file__).resolve().parent.parent / "m64_pre.sha256"

DS_ORDER = ["training_run", "workflow", "evaluation", "comparison",
            "suite_run", "tokenizer_training", "tokenized_version"]
TOK_ORDER = ["training_run", "workflow", "evaluation", "comparison",
             "suite_run", "sample", "sample_quality", "tokenized_dataset"]

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    results.append((name, bool(cond), extra))
    print(f"{'PASS' if cond else 'FAIL'} {name} {extra}")


def jget(path: str):
    with urllib.request.urlopen(API + path, timeout=30) as r:
        assert r.status == 200, (path, r.status)
        return json.loads(r.read()), r.read()


def inventory(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.relative_to(root).parts[0] != "tmp":
            out[p.relative_to(root).as_posix()] = \
                hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ---------------------- the INDEPENDENT oracle (no app code) ---------------- #
def oracle(root: Path) -> tuple[dict, dict]:
    """Recompute both usage overviews from the raw manifests."""
    ds: dict[str, dict[str, list[str]]] = {}
    tok: dict[str, dict[str, list[str]]] = {}

    def buckets(kind_id, order):
        return {kind_id: {c: [] for c in order}}

    ds.update(buckets(DS_ID, DS_ORDER))
    tok.update(buckets(TOK_ID, TOK_ORDER))

    mroot = root / "models"
    model_ids = sorted(p.name for p in mroot.iterdir()
                       if p.is_dir() and (p / "manifest.json").exists())
    for mid in model_ids:
        man = json.loads((mroot / mid / "manifest.json").read_text())
        for prov in man.get("training_provenance", []):
            if prov.get("dataset_id") == DS_ID:
                ds[DS_ID]["training_run"].append(f"{mid}/{prov['run_id']}")
            if prov.get("tokenizer_id") == TOK_ID:
                tok[TOK_ID]["training_run"].append(f"{mid}/{prov['run_id']}")
        for fam, key, idfield in (("evaluations", "evaluation", "eval_id"),
                                  ("comparisons", "comparison",
                                   "comparison_id")):
            froot = mroot / mid / fam
            if not froot.exists():
                continue
            for d in sorted(froot.iterdir()):
                if not d.is_dir():
                    continue
                try:
                    rec = json.loads((d / "manifest.json").read_text())
                except Exception:
                    continue
                if rec.get("dataset_id") == DS_ID:
                    ds[DS_ID][key].append(f"{mid}/{rec[idfield]}")
                if rec.get("tokenizer_id") == TOK_ID:
                    tok[TOK_ID][key].append(f"{mid}/{rec[idfield]}")
        wroot = mroot / mid / "workflows"
        if wroot.exists():
            for d in sorted(wroot.iterdir()):
                if not d.is_dir():
                    continue
                try:
                    rec = json.loads((d / "manifest.json").read_text())
                except Exception:
                    continue
                pairs = set()
                for stage in rec.get("plan", {}).get("stages", []):
                    tr = stage.get("training")
                    if tr:
                        pairs.add((tr.get("dataset_id"),
                                   tr.get("tokenizer_id")))
                    ev = stage.get("evaluation")
                    if ev:
                        cfg = ev.get("config", {})
                        pairs.add((cfg.get("dataset_id"),
                                   cfg.get("tokenizer_id")))
                    cp = stage.get("comparison")
                    if cp:
                        pairs.add((cp.get("dataset_id"),
                                   cp.get("tokenizer_id")))
                if any(p[0] == DS_ID for p in pairs):
                    ds[DS_ID]["workflow"].append(
                        f"{mid}/{rec['workflow_id']}")
                if any(p[1] == TOK_ID for p in pairs):
                    tok[TOK_ID]["workflow"].append(
                        f"{mid}/{rec['workflow_id']}")
    # root-level families
    sroot = root / "suite-runs"
    if sroot.exists():
        for d in sorted(sroot.iterdir()):
            if not d.is_dir():
                continue
            try:
                rec = json.loads((d / "manifest.json").read_text())
            except Exception:
                continue
            mid = rec.get("model_id")
            probes = [r.get("probe", {}) for r in rec.get("results", [])]
            if any(p.get("dataset_id") == DS_ID for p in probes):
                ds[DS_ID]["suite_run"].append(f"{mid}/{rec['suite_run_id']}")
            if any(p.get("tokenizer_id") == TOK_ID for p in probes):
                tok[TOK_ID]["suite_run"].append(f"{mid}/{rec['suite_run_id']}")
    for fam, key, idfield in (("samples", "sample", "sample_id"),
                              ("sample-evaluations", "sample_quality",
                               "evaluation_id")):
        froot = root / fam
        if not froot.exists():
            continue
        for mdir in sorted(froot.iterdir()):
            if not mdir.is_dir():
                continue
            for d in sorted(mdir.iterdir()):
                if not d.is_dir():
                    continue
                try:
                    rec = json.loads((d / "manifest.json").read_text())
                except Exception:
                    continue
                if rec.get("tokenizer_id") == TOK_ID:
                    tok[TOK_ID][key].append(f"{mdir.name}/{rec[idfield]}")
    # tokenizer training sources
    troot = root / "tokenizers"
    for d in sorted(troot.iterdir()):
        if not d.is_dir():
            continue
        try:
            rec = json.loads((d / "manifest.json").read_text())
        except Exception:
            continue
        if rec.get("trained_on_dataset_id") == DS_ID:
            ds[DS_ID]["tokenizer_training"].append(d.name)
    # tokenized layout (both directions)
    droot = root / "datasets"
    for d in sorted(droot.iterdir()):
        if not d.is_dir():
            continue
        for vdir in sorted(d.iterdir()):
            if not (vdir.is_dir() and vdir.name.startswith("v")):
                continue
            troot2 = vdir / "tokenized"
            if not troot2.exists():
                continue
            for tdir in sorted(troot2.iterdir()):
                if not tdir.is_dir():
                    continue
                if d.name == DS_ID:
                    ds[DS_ID]["tokenized_version"].append(
                        f"{vdir.name}/{tdir.name}")
                if tdir.name == TOK_ID:
                    tok[TOK_ID]["tokenized_dataset"].append(
                        f"{d.name}/{vdir.name}")
    for b in (ds, tok):
        for k in b:
            for c in b[k]:
                b[k][c] = sorted(b[k][c])
    return ds, tok


# ============================ prelude (read-only) ========================== #
project, _ = jget("/project")
root = Path(project["storage_root"])
check("P1 project readable", project["model_count"] == 1)

pre_disk = inventory(root)
pre_ref = {}
for line in PRE_INVENTORY.read_text().splitlines():
    digest, rel = line.split("  ", 1)
    pre_ref[rel.removeprefix("./")] = digest
check("P2 disk == m64_pre.sha256 (52/52)", pre_disk == pre_ref,
      f"{len(pre_ref)} files")

datasets, _ = jget("/datasets")
check("P3 dataset registry", [d["id"] for d in datasets] == [DS_ID])
tokenizers, _ = jget("/tokenizers")
check("P4 tokenizer registry", [t["id"] for t in tokenizers] == [TOK_ID])

ret, _ = jget(f"/models/{MODEL_ID}/checkpoints/retention")
check("P5 M62 retention coherent (cross-milestone)",
      ret["total_checkpoints"] == N_CKPTS
      and ret["deletable_checkpoints"] == N_DELETABLE
      and ret["protected_checkpoints"] == N_PROTECTED,
      f"{ret['total_checkpoints']}/{ret['deletable_checkpoints']}/"
      f"{ret['protected_checkpoints']}")

sto, _ = jget("/project/storage")
check("P6 M63 project storage coherent (cross-milestone)",
      sto["total_files"] == 52 and sto["model_count"] == 1)

ds_oracle, tok_oracle = oracle(root)

# ===================== the M64 overviews (once each) ======================= #
ov, body_ds = jget(f"/datasets/{DS_ID}/usage")
check("X1 dataset usage readable", ov["dataset_id"] == DS_ID,
      f"total={ov['total_references']}")

# ---- V1: identity ---- #
ds_meta, _ = jget(f"/datasets/{DS_ID}")
check("V1 dataset identity fields",
      ov["name"] == DS_NAME == ds_meta["dataset"]["name"]
      and ov["version_count"] == len(ds_meta["dataset"]["versions"]) == 1
      and ov["latest_version"] == ds_meta["dataset"]["latest_version"] == 1
      and ov["created_at"] == ds_meta["dataset"]["created_at"])

# ---- V2: canonical order + per-category oracle parity + counts ---- #
cats = {c["category"]: c["references"] for c in ov["categories"]}
check("V2 canonical category order", [c["category"] for c in
                                      ov["categories"]] == DS_ORDER)
ok_v3 = True
for c in DS_ORDER:
    ok_v3 = ok_v3 and cats[c] == ds_oracle[DS_ID][c] \
        and len(cats[c]) == DS_CATS[c]
check("V3 every dataset category == independent oracle + certified counts",
      ok_v3,
      " ".join(f"{c}={len(cats[c])}" for c in DS_ORDER))

# ---- V4: evaluation/comparison parity with the PUBLIC routes ---- #
evals, _ = jget(f"/models/{MODEL_ID}/evaluations/by-dataset/{DS_ID}")
comps, _ = jget(f"/models/{MODEL_ID}/comparisons/by-dataset/{DS_ID}")
check("V4 dataset categories == public cross-reference routes",
      cats["evaluation"] == sorted(
          f"{MODEL_ID}/{e['eval_id']}" for e in evals)
      and cats["comparison"] == sorted(
          f"{MODEL_ID}/{c['comparison_id']}" for c in comps),
      f"evals={len(evals)} comps={len(comps)}")

# ---- V5: the guard category is exactly the deletion guard's list ---- #
check("V5 tokenizer_training == the ONE guard scan",
      cats["tokenizer_training"] == [TOK_ID])

# ---- V6: totals ---- #
check("V6 dataset totals",
      ov["total_references"] == DS_TOTAL
      == sum(len(v) for v in cats.values())
      and ov["referenced"] is True)

tv, body_tok = jget(f"/tokenizers/{TOK_ID}/usage")
check("X2 tokenizer usage readable", tv["tokenizer_id"] == TOK_ID,
      f"total={tv['total_references']}")

tok_meta, _ = jget(f"/tokenizers/{TOK_ID}")
check("V7 tokenizer identity fields",
      tv["name"] == TOK_NAME == tok_meta["name"]
      and tv["trained_on_dataset_id"] == DS_ID
      == tok_meta["trained_on_dataset_id"]
      and tv["requested_vocab_size"] == tok_meta["requested_vocab_size"]
      and tv["actual_vocab_size"] == tok_meta["actual_vocab_size"])

tcats = {c["category"]: c["references"] for c in tv["categories"]}
check("V8 canonical category order",
      [c["category"] for c in tv["categories"]] == TOK_ORDER)
ok_v9 = True
for c in TOK_ORDER:
    ok_v9 = ok_v9 and tcats[c] == tok_oracle[TOK_ID][c] \
        and len(tcats[c]) == TOK_CATS[c]
check("V9 every tokenizer category == independent oracle + certified counts",
      ok_v9,
      " ".join(f"{c}={len(tcats[c])}" for c in TOK_ORDER))

evals_t, _ = jget(f"/models/{MODEL_ID}/evaluations/by-tokenizer/{TOK_ID}")
comps_t, _ = jget(f"/models/{MODEL_ID}/comparisons/by-tokenizer/{TOK_ID}")
check("V10 tokenizer categories == public cross-reference routes",
      tcats["evaluation"] == sorted(
          f"{MODEL_ID}/{e['eval_id']}" for e in evals_t)
      and tcats["comparison"] == sorted(
          f"{MODEL_ID}/{c['comparison_id']}" for c in comps_t))

check("V11 tokenizer totals",
      tv["total_references"] == TOK_TOTAL
      == sum(len(v) for v in tcats.values())
      and tv["referenced"] is True)

# ---- V12: shared categories agree between the two views ---- #
check("V12 shared categories agree across the two views",
      all(cats[c] == tcats[c] for c in
          ("training_run", "workflow", "evaluation", "comparison",
           "suite_run")))

# ---- V13: unknown ids -> 404 ---- #
import urllib.error
for path in ("/datasets/no-such-m64/usage",
             "/tokenizers/no-such-m64/usage"):
    try:
        urllib.request.urlopen(API + path, timeout=30)
        ok404 = False
    except urllib.error.HTTPError as e:
        ok404 = e.code == 404
    check(f"V13 unknown -> 404 ({path.rsplit('/', 2)[-2]})", ok404)

# ---- V14: deterministic byte-identical repeats ---- #
_, b2 = jget(f"/datasets/{DS_ID}/usage")
_, t2 = jget(f"/tokenizers/{TOK_ID}/usage")
check("V14 deterministic byte-identical repeats",
      b2 == body_ds and t2 == body_tok)

# ============================== audits ===================================== #
post_disk = inventory(root)
check("A1 storage byte-identical before/after (zero mutation)",
      post_disk == pre_disk, f"{len(post_disk)} files")
check("A2 tmp clean", not list((root / "tmp").iterdir()))

n_pass = sum(1 for _, ok, _ in results if ok)
print()
print(f"{n_pass}/{len(results)} LIVE M64 READ-ONLY CHECKS PASSED "
      "(zero mutations)")
sys.exit(0 if n_pass == len(results) else 1)
