"""M16 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M16 writes are strictly limited to ONE
sample-evaluation manifest per successful quality request under the new
root-level ``sample-evaluations/`` family — exactly the accounting the Task
prescribes.

Production facts (audited at Phase A): model 4a0a871886ef (config vocab 640,
context 64), checkpoint 0511de4c7372, the ONLY stored tokenizer
99106e3255c5 (actual vocab 320 <= model vocab, the platform compatibility
convention), 4 immutable M15 samples under samples/4a0a871886ef/, 16 M4
evaluations, 13 workflows, 10 suite-runs, 7 recipes — 94 files /
4,000,221 B / 0 .tmp.

Phase A (baseline audit): exact storage audit (94 files / 4,000,221 B /
0 .tmp — the M15 certified handoff), full per-file hash inventory saved to
/tmp/m16-smoke-baseline-inventory.json, model/checkpoint/tokenizer
existence over the API, the four M15 sample manifests verified present (a
sample is SELECTED by listing production samples — never hard-coded before
verification), empty sample-evaluations history, M4 evaluation / workflow /
suite-run counts and a deterministic dashboard baseline.

Phase B (successful measurement of one discovered production sample): POST
/models/4a0a871886ef/samples/<discovered>/quality (empty body) -> 200 with
sample-driven state resolution (checkpoint 0511de4c7372 + tokenizer
99106e3255c5 resolved from the manifest and verified), exact metrics
(loss_nats / perplexity) and generated-target bookkeeping
(evaluated == generated == the sample's recorded count, prompt tokens
condition only, window within the 64-token context, window_rule
single_window). The metric is INDEPENDENTLY recomputed in-process from the
verified checkpoint bytes (first-principles causal CE over exactly the
generated target positions) and must match the server's rounded values.
An identical repeat request creates a second immutable manifest with
byte-identical result_hash/metrics and a distinct evaluation_id. Sample,
checkpoint, tokenizer and M4-evaluation artifacts are byte-untouched;
exactly one manifest per request under sample-evaluations/.

Phase C (live failure paths, zero storage change): unknown model -> 404;
unknown sample -> 404; a sample id requested under the wrong (other
production) model -> 404; a fabricated sample whose recorded checkpoint
does not exist -> 404 (missing dependency, safely exercised on a transient
copy); an overlong fabricated sample -> 422; a tampered fabricated sample
(result_hash non-reproduction) -> 409. Every fabricated sample is created
on a transient directory and removed in ``finally``; NO production artifact
is ever corrupted or modified.

Phase D (final audit): inventory diff vs Phase A — every pre-existing file
byte-identical; new files EXACTLY the 2 sample-evaluation manifests;
totals 96 files / 0 .tmp; the only new storage family is
sample-evaluations/; samples (4) and M4 evaluations (16) unchanged;
sample-quality list/get deterministic on repeat; dashboard deterministic
with the same result_hash and no diagnostics.

Exit code 0 = all checks passed.

Re-run note: the smoke certifies against the documented M16 baseline
(94 / 4,000,221 / 0, sample-evaluations/ absent). Re-running it after a
successful run requires restoring that baseline first (remove the
sample-evaluations/ family the run created) — the phases assert the exact
pre-state, like the M13–M15 smokes.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8741/api/v1"
MODEL = "4a0a871886ef"          # production model with the full history
OTHER_MODEL = "b5bc905326b6"    # second production model (no samples)
CKPT = "0511de4c7372"           # its kept checkpoint (suite evidence)
TOK = "99106e3255c5"            # the ONLY stored tokenizer (vocab 320)
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m16-smoke-baseline-inventory.json")

PRE_FILES, PRE_BYTES, PRE_TMP = 94, 4_000_221, 0
CTX = 64

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def call(method: str, path: str, body=None):
    req = urllib.request.Request(BASE + path, method=method)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, data=data) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def call_json(method: str, path: str, body=None):
    code, raw = call(method, path, body)
    return code, json.loads(raw) if raw else None


def audit(label: str) -> tuple[int, int, int, dict]:
    files = [p for p in ROOT.rglob("*") if p.is_file()]
    n_bytes = sum(p.stat().st_size for p in files)
    n_tmp = sum(1 for p in files if ".tmp" in p.name)
    snap = {p.relative_to(ROOT).as_posix():
            hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    print(f"    storage: {len(files)} files / {n_bytes} B / tmp {n_tmp} "
          f"({label})")
    return len(files), n_bytes, n_tmp, snap


def quality_url(model_id: str, sample_id: str) -> str:
    return f"/models/{model_id}/samples/{sample_id}/quality"


def independent_loss(sample: dict, weights_sha: str) -> tuple[float, float]:
    """First-principles recomputation of the measured metric from the
    verified production checkpoint bytes (read-only, no writes).

    Rebuilds the model from the stored config, restores the checkpoint
    state, forwards the recorded prompt+generated ids ONCE and computes the
    causal cross-entropy over exactly the generated-target positions
    (positions p_len-1 .. T-2) — the same M4 objective the server uses.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    from app.model_builder import (build_transformer, content_hash,
                                   restore_state)
    from app.storage import Storage
    from app.training import TrainingEngine

    storage = Storage(ROOT)
    training = TrainingEngine(storage)
    state = training.verify_checkpoint(MODEL, sample["checkpoint_id"])
    assert content_hash(state) == weights_sha
    model_cfg = storage.load_record(MODEL).config
    model = build_transformer(model_cfg).module
    missing, unexpected = restore_state(model, state)
    assert not missing and not unexpected
    model.eval()
    ids = sample["prompt_token_ids"] + sample["generated_token_ids"]
    p_len = len(sample["prompt_token_ids"])
    g_len = len(sample["generated_token_ids"])
    with torch.no_grad():
        logits, _ = model(torch.tensor([ids], dtype=torch.long))
    logits = logits[0].float()
    score = logits[p_len - 1: p_len + g_len - 1]      # (G, V)
    targets = torch.tensor(ids, dtype=torch.long)[p_len:]
    raw = float(F.cross_entropy(score, targets))      # mean over G targets
    return round(raw, 6), round(float(np.exp(min(raw, 100.0))), 6)


def main() -> int:
    print("== Phase A: baseline audit + saved inventory ==")
    pre_n, pre_bytes, pre_tmp, pre_snap = audit("pre")
    check("A1 exact baseline audit 94/4,000,221/0",
          (pre_n, pre_bytes, pre_tmp) == (PRE_FILES, PRE_BYTES, PRE_TMP),
          f"{pre_n}/{pre_bytes}/{pre_tmp}")
    INVENTORY.write_text(json.dumps(pre_snap, indent=1, sort_keys=True))
    check("A2 full per-file hash inventory saved",
          INVENTORY.exists() and len(json.loads(INVENTORY.read_text()))
          == pre_n, str(INVENTORY))
    code, m = call_json("GET", f"/models/{MODEL}")
    check("A3 model exists with vocab 640 / context 64",
          code == 200 and m["config"]["vocab_size"] == 640
          and m["config"]["context_length"] == CTX, f"{code}")
    code, c = call_json("GET", f"/models/{MODEL}/checkpoints/{CKPT}")
    check("A4 checkpoint exists + weights_sha256 recorded",
          code == 200 and len(c["weights_sha256"]) == 64)
    code, tok = call_json("GET", "/tokenizers")
    ids = [x["id"] for x in tok] if isinstance(tok, list) else []
    check("A5 tokenizer 99106e3255c5 present, actual vocab <= 640",
          code == 200 and TOK in ids and
          next(x["actual_vocab_size"] for x in tok if x["id"] == TOK) <= 640)
    code, smp = call_json("GET", f"/models/{MODEL}/samples")
    check("A6 exactly 4 M15 samples present (none hard-coded unseen)",
          code == 200 and len(smp) == 4 and len({s["sample_id"]
                                                 for s in smp}) == 4)
    sample = sorted(smp, key=lambda s: (s["created_at"], s["sample_id"]))[0]
    check("A7 selected sample verified from the live listing",
          sample["model_id"] == MODEL
          and sample["checkpoint_id"] == CKPT
          and sample["tokenizer_id"] == TOK
          and sample["generated_token_count"] >= 1
          and all(0 <= i < 320 for i in sample["generated_token_ids"]),
          f"sample {sample['sample_id']}")
    code, ev0 = call_json("GET", f"/models/{MODEL}/sample-quality")
    check("A8 sample-evaluations history empty",
          code == 200 and ev0 == []
          and not (ROOT / "sample-evaluations").exists())
    code, dash0 = call_json("GET", f"/models/{MODEL}/dashboard")
    check("A9 dashboard baseline deterministic",
          code == 200 and dash0["result_hash"] == call_json(
              "GET", f"/models/{MODEL}/dashboard")[1]["result_hash"])
    check("A10 M4 evaluations / workflows / suite-runs / recipes counts",
          len(ev0_evals := call_json(
              "GET", f"/models/{MODEL}/evaluations")[1]) == 16
          and len(call_json("GET", f"/models/{MODEL}/workflows")[1]) == 13
          and dash0["suite_runs"]["counts"] == {"completed": 10}
          and len(call_json("GET", "/workflows/recipes")[1]) == 7,
          f"evals={len(ev0_evals)}")

    print("== Phase B: measure one discovered production sample ==")
    code, q1 = call_json("POST", quality_url(MODEL, sample["sample_id"]))
    e1 = q1
    check("B1 quality POST -> 200 with sample-driven state resolution",
          code == 200 and e1["model_id"] == MODEL
          and e1["sample_id"] == sample["sample_id"]
          and e1["sample_result_hash"] == sample["result_hash"]
          and e1["checkpoint_id"] == CKPT
          and e1["checkpoint_weights_sha256"] == c["weights_sha256"]
          and e1["tokenizer_id"] == TOK, f"{code}")
    g = sample["generated_token_count"]
    p = sample["prompt_token_count"]
    check("B2 exact generated-target bookkeeping (prompt never scored)",
          e1["evaluated_token_count"] == g == e1["generated_token_count"]
          and e1["prompt_token_count"] == p
          and e1["window_token_count"] == p + g <= CTX
          and e1["context_length"] == CTX
          and e1["window_rule"] == "single_window"
          and len(e1["token_sequence_sha256"]) == 64,
          f"P={p} G={g} window={e1['window_token_count']}/{CTX}")
    loss_ok = math.isfinite(e1["loss_nats"]) and e1["loss_nats"] >= 0.0
    ppl_env = abs(e1["perplexity"]
                  - math.exp(min(e1["loss_nats"], 100.0))) < 1e-3
    check("B3 metric fields sane (M4 perplexity convention)",
          loss_ok and ppl_env and len(e1["result_hash"]) == 64,
          f"loss={e1['loss_nats']} ppl={e1['perplexity']}")
    i_loss, i_ppl = independent_loss(sample, c["weights_sha256"])
    check("B4 metric independently reproduced from the verified bytes",
          i_loss == e1["loss_nats"] and abs(i_ppl - e1["perplexity"]) < 1e-3,
          f"independent {i_loss}/{i_ppl} vs record "
          f"{e1['loss_nats']}/{e1['perplexity']}")
    code, q2 = call_json("POST", quality_url(MODEL, sample["sample_id"]))
    e2 = q2
    check("B5 identical repeat: same metrics+hash, distinct evaluation_id",
          code == 200 and e2["evaluation_id"] != e1["evaluation_id"]
          and e2["loss_nats"] == e1["loss_nats"]
          and e2["perplexity"] == e1["perplexity"]
          and e2["result_hash"] == e1["result_hash"], f"{code}")
    code, lst = call_json("GET", f"/models/{MODEL}/sample-quality")
    keyed = [(x["created_at"], x["evaluation_id"]) for x in lst]
    check("B6 list: 2 records in (created_at, evaluation_id) order",
          code == 200 and len(lst) == 2 and keyed == sorted(keyed))
    check("B7 list/get byte-deterministic on repeat",
          call("GET", f"/models/{MODEL}/sample-quality")[1] ==
          call("GET", f"/models/{MODEL}/sample-quality")[1]
          and call("GET", f"/models/{MODEL}/sample-quality/"
                          f"{e1['evaluation_id']}")[1] ==
          call("GET", f"/models/{MODEL}/sample-quality/"
                      f"{e1['evaluation_id']}")[1])
    one = call_json("GET", f"/models/{MODEL}/sample-quality/"
                           f"{e1['evaluation_id']}")
    check("B8 get returns the persisted record (result_hash intact)",
          one[0] == 200 and one[1]["result_hash"] == e1["result_hash"])
    # storage: exactly the two new manifests under sample-evaluations/
    ev_dir = ROOT / "sample-evaluations" / MODEL
    manis = sorted(p.parent.name
                   for p in ev_dir.glob("evaluation-*/manifest.json"))
    check("B9 exactly one manifest per successful request",
          len(manis) == 2
          and {f"evaluation-{e['evaluation_id']}" for e in (e1, e2)}
          == set(manis))
    check("B10 sample/checkpoint/tokenizer/M4 artifacts untouched",
          call_json("GET", f"/models/{MODEL}/samples/{sample['sample_id']}")
          [1]["result_hash"] == sample["result_hash"]
          and len(call_json("GET", f"/models/{MODEL}/evaluations")[1]) == 16
          and len(call_json("GET", f"/models/{MODEL}/samples")[1]) == 4)

    print("== Phase C: live failure paths (zero storage change) ==")
    n_before = len(call_json("GET", f"/models/{MODEL}/sample-quality")[1])
    real_manifest = json.loads(call("GET", f"/models/{MODEL}/samples/"
                                           f"{sample['sample_id']}")[1])
    transient: list[Path] = []
    try:
        # C1-C4: plain resolution 404s
        check("C1 unknown model -> 404",
              call_json("POST", quality_url("no-such-model",
                                            sample["sample_id"]))[0] == 404)
        check("C2 unknown sample -> 404",
              call_json("POST", quality_url(MODEL, "no-such-sample"))[0]
              == 404)
        check("C3 sample id under the WRONG model -> 404",
              call_json("POST", quality_url(OTHER_MODEL,
                                            sample["sample_id"]))[0] == 404)
        # C5: fabricated sample recording a MISSING checkpoint (transient)
        fake = dict(real_manifest, sample_id="m16missing-ckpt")
        fake["checkpoint_id"] = "no-such-ckpt"
        d = ROOT / "samples" / MODEL / "sample-m16missing-ckpt"
        d.mkdir(parents=True)
        transient.append(d)
        (d / "manifest.json").write_text(json.dumps(fake))
        check("C5 recorded checkpoint missing -> 404",
              call_json("POST", quality_url(MODEL,
                                            "m16missing-ckpt"))[0] == 404)
        # C6: fabricated OVERLONG sample -> 422 (never silently truncated)
        ids = real_manifest["generated_token_ids"]
        over = dict(real_manifest, sample_id="m16overlong",
                    generated_token_ids=(ids * 17)[:100],
                    generated_token_count=100)
        d = ROOT / "samples" / MODEL / "sample-m16overlong"
        d.mkdir(parents=True)
        transient.append(d)
        (d / "manifest.json").write_text(json.dumps(over))
        check("C6 overlong sample -> 422",
              call_json("POST", quality_url(MODEL, "m16overlong"))[0] == 422)
        # C7: fabricated TAMPERED sample (result_hash broken) -> 409
        x0 = ids[0]
        tam = dict(real_manifest, sample_id="m16tampered",
                   generated_token_ids=[
                       (x0 - 1 if x0 > 0 else 1) if i == 0 else x
                       for i, x in enumerate(ids)])
        d = ROOT / "samples" / MODEL / "sample-m16tampered"
        d.mkdir(parents=True)
        transient.append(d)
        (d / "manifest.json").write_text(json.dumps(tam))
        check("C7 tampered sample (result_hash non-reproduction) -> 409",
              call_json("POST", quality_url(MODEL, "m16tampered"))[0] == 409)
        # C8: GET semantics of the read-only endpoints stay 404-clean
        check("C8 unknown model/eval on read endpoints -> 404",
              call_json("GET", "/models/no-such-model/sample-quality")[0]
              == 404
              and call_json("GET", f"/models/{MODEL}/sample-quality/no-such")
              [0] == 404)
    finally:
        for d in transient:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
    check("C9 zero sample-evaluation growth after every failure",
          len(call_json("GET", f"/models/{MODEL}/sample-quality")[1])
          == n_before)
    code, _ = call_json("GET", f"/models/{MODEL}/samples")
    check("C10 sample count unchanged (4) after transient crafts",
          code == 200 and len(_) == 4)

    print("== Phase D: final audit (historical files byte-identical) ==")
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    expected = {f"sample-evaluations/{MODEL}/evaluation-{e['evaluation_id']}"
                f"/manifest.json" for e in (e1, e2)}
    check("D1 every pre-existing file byte-identical", changed == [],
          f"{len(changed)} changed")
    check("D2 new files are EXACTLY the 2 sample-evaluation manifests",
          new_files == expected, f"{len(new_files)} new")
    check("D3 totals 96 files / zero .tmp / only the M16 family added",
          f_n == 96 and f_tmp == 0
          and all(k.startswith("sample-evaluations/") for k in new_files)
          and (ROOT / "sample-evaluations").is_dir()
          and len(list((ROOT / "sample-evaluations" / MODEL)
                       .glob("evaluation-*"))) == 2)
    dash1 = call_json("GET", f"/models/{MODEL}/dashboard")[1]
    check("D4 dashboard still deterministic, healthy, hash unchanged",
          dash1 == call_json("GET", f"/models/{MODEL}/dashboard")[1]
          and dash1["result_hash"] == dash0["result_hash"]
          and dash1["diagnostics"] == [])
    smp_after = call_json("GET", f"/models/{MODEL}/samples")[1]
    check("D5 samples and M4 evaluations unchanged (4 / 16)",
          len(smp_after) == 4
          and len(call_json("GET", f"/models/{MODEL}/evaluations")[1]) == 16)

    print()
    if FAILURES:
        print(f"M16 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M16 live smoke OK: per-sample causal-LM quality measurement "
          "(sample-driven state resolution, exact generated-target "
          "accounting, independent metric reproduction, immutable "
          "sample-evaluation records, exact storage accounting) verified "
          "live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
