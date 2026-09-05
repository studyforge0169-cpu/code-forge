"""M15 live HTTP smoke against the production FORGE_ROOT
(/home/user/ai-model-forge-data). M15 writes are strictly limited to ONE
sample manifest per successful generation request under the new root-level
``samples/`` family — exactly the accounting the Task prescribes.

Production facts (audited at Phase A): model 4a0a871886ef (config
vocab_size 640, context 64), checkpoint 0511de4c7372 (the kept checkpoint
with the full suite history), the ONLY stored tokenizer 99106e3255c5
(actual vocab 320 — <= the model vocab, the platform compatibility
convention M3/M4 document and M15 reuses; logits are restricted to the
tokenizer's vocab so every generated token decodes).

Phase A (baseline audit): exact storage audit (90 files / 3,994,597 B /
0 .tmp — the M14 handoff), full per-file hash inventory saved to
/tmp/m15-smoke-baseline-inventory.json, model/checkpoint/tokenizer
existence over the API, empty samples history for both models.

Phase B (successful generation on 4a0a871886ef + 0511de4c7372): one greedy
request and its identical repeat (distinct sample ids, identical token ids
+ output text + result_hash), one seeded temperature request and its
identical repeat; every response carries the verified checkpoint weights
hash + tokenizer content hash; exactly one new sample manifest per request;
list ordering (created_at, sample_id) and byte-identical repeated GETs;
get works; the 16 evaluations / 13 workflows / 10 suite-runs / 7 recipes
are untouched.

Phase C (live failure paths): unknown checkpoint / tokenizer / model ->
404; strategy violations (greedy+seed, temperature w/o seed or temperature,
temperature=0) -> 422; max_new_tokens 0 / 513 / context overflow -> 422;
empty prompt -> 422 — every failure with zero storage change.

Phase D (final audit): inventory diff vs Phase A — every pre-existing file
byte-identical; new files EXACTLY the 4 sample manifests; totals 94 files /
zero .tmp; samples list/get deterministic on repeat; the only new storage
family is samples/.

Exit code 0 = all checks passed.

Re-run note: the smoke certifies against the documented M15 baseline (90 /
3,994,597 / 0, samples/ absent). Re-running it after a successful run
requires restoring that baseline first (remove the samples/ family the run
created) — the phases assert the exact pre-state, like the M13/M14 smokes.
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8741/api/v1"
MODEL = "4a0a871886ef"          # production model with the full history
CKPT = "0511de4c7372"           # its kept checkpoint (suite evidence)
TOK = "99106e3255c5"            # the ONLY stored tokenizer (vocab 320)
ROOT = Path("/home/user/ai-model-forge-data")
INVENTORY = Path("/tmp/m15-smoke-baseline-inventory.json")

PRE_FILES, PRE_BYTES, PRE_TMP = 90, 3_994_597, 0
PROMPT = "river mountain cloud forest ocean desert valley island"
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


def gen_body(**overrides) -> dict:
    body = {"model_id": MODEL, "checkpoint_id": CKPT, "tokenizer_id": TOK,
            "prompt": PROMPT, "strategy": "greedy", "max_new_tokens": 8}
    body.update(overrides)
    return body


def main() -> int:
    print("== Phase A: baseline audit + saved inventory ==")
    pre_n, pre_bytes, pre_tmp, pre_snap = audit("pre")
    check("A1 exact baseline audit 90/3,994,597/0",
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
    check("A6 samples history empty for both models",
          code == 200 and smp == []
          and call_json("GET", "/models/b5bc905326b6/samples")[1] == [])
    code, dash0 = call_json("GET", f"/models/{MODEL}/dashboard")
    check("A7 dashboard baseline deterministic",
          code == 200 and dash0["result_hash"] == call_json(
              "GET", f"/models/{MODEL}/dashboard")[1]["result_hash"])
    _, evals0 = call_json("GET", f"/models/{MODEL}/evaluations")
    check("A8 16 evaluations / 13 workflows / 10 suite-runs pre-state",
          len(evals0) == 16
          and dash0["workflows"]["counts"] ==
          {"completed": 9, "failed": 3, "stopped": 1}
          and dash0["suite_runs"]["counts"] == {"completed": 10})

    print("== Phase B: successful generation (one manifest per request) ==")
    g1 = call_json("POST", "/samples/generate", gen_body())
    check("B1 greedy run 1 -> 200, qualified record",
          g1[0] == 200 and g1[1]["strategy"] == "greedy"
          and g1[1]["model_id"] == MODEL and g1[1]["checkpoint_id"] == CKPT
          and g1[1]["tokenizer_id"] == TOK, f"{g1[0]}")
    s1 = g1[1]
    g2 = call_json("POST", "/samples/generate", gen_body())
    s2 = g2[1]
    check("B2 identical greedy repeat: same ids/text/hash, distinct id",
          g2[0] == 200 and s2["sample_id"] != s1["sample_id"]
          and s2["generated_token_ids"] == s1["generated_token_ids"]
          and s2["output_text"] == s1["output_text"]
          and s2["result_hash"] == s1["result_hash"], f"{g2[0]}")
    check("B3 audit fields: verified weights hash + tokenizer hash",
          s1["checkpoint_weights_sha256"] == c["weights_sha256"]
          and len(s1["tokenizer_hash"]) == 64
          and s1["generated_token_count"] == 8
          and all(0 <= i < 320 for i in s1["generated_token_ids"]))
    t1 = call_json("POST", "/samples/generate",
                   gen_body(strategy="temperature", temperature=0.9, seed=21))
    t2 = call_json("POST", "/samples/generate",
                   gen_body(strategy="temperature", temperature=0.9, seed=21))
    check("B4 seeded temperature repeat deterministic",
          t1[0] == 200 and t2[0] == 200
          and t1[1]["sample_id"] != t2[1]["sample_id"]
          and t1[1]["generated_token_ids"] == t2[1]["generated_token_ids"]
          and t1[1]["result_hash"] == t2[1]["result_hash"], f"{t1[0]}")
    code, lst = call_json("GET", f"/models/{MODEL}/samples")
    keyed = [(r["created_at"], r["sample_id"]) for r in lst]
    check("B5 list: 4 samples in (created_at, sample_id) order",
          code == 200 and len(lst) == 4 and keyed == sorted(keyed))
    check("B6 list/get byte-deterministic on repeat",
          call("GET", f"/models/{MODEL}/samples")[1] ==
          call("GET", f"/models/{MODEL}/samples")[1]
          and call("GET", f"/models/{MODEL}/samples/{s1['sample_id']}")[1]
          == call("GET", f"/models/{MODEL}/samples/{s1['sample_id']}")[1])
    one = call_json("GET", f"/models/{MODEL}/samples/{s1['sample_id']}")
    check("B7 get returns the persisted record (result_hash intact)",
          one[0] == 200 and one[1]["result_hash"] == s1["result_hash"]
          and one[1]["prompt"] == PROMPT)
    check("B8 unknown sample/model -> 404",
          call_json("GET", f"/models/{MODEL}/samples/no-such")[0] == 404
          and call_json("GET", "/models/no-such-model/samples")[0] == 404)
    _, evals1 = call_json("GET", f"/models/{MODEL}/evaluations")
    check("B9 no evaluation/checkpoint/model changes",
          len(evals1) == 16
          and call_json("GET", f"/models/{MODEL}/workflows")[1]
          and len(call_json("GET", f"/models/{MODEL}/workflows")[1]) == 13)

    print("== Phase C: live failure paths (zero storage change) ==")
    n_before = len(call_json("GET", f"/models/{MODEL}/samples")[1])
    failures = [
        ("unknown checkpoint", gen_body(checkpoint_id="no-such-ckpt"), 404),
        ("unknown tokenizer", gen_body(tokenizer_id="no-such-tok"), 404),
        ("unknown model", gen_body(model_id="no-such-model"), 404),
        ("greedy + seed", gen_body(strategy="greedy", seed=1), 422),
        ("temperature w/o seed", gen_body(strategy="temperature",
                                          temperature=0.9), 422),
        ("temperature w/o temperature", gen_body(strategy="temperature",
                                                 seed=1), 422),
        ("temperature=0", gen_body(strategy="temperature", temperature=0.0,
                                   seed=1), 422),
        ("max_new_tokens=0", gen_body(max_new_tokens=0), 422),
        ("max_new_tokens=513", gen_body(max_new_tokens=513), 422),
        ("context overflow", gen_body(max_new_tokens=60), 422),
        ("empty prompt", gen_body(prompt=""), 422),
    ]
    for i, (label, body, want) in enumerate(failures, start=1):
        code, _ = call_json("POST", "/samples/generate", body)
        check(f"C{i} {label} -> {want}", code == want, f"http {code}")
    check("C12 no new sample manifest from any failure",
          len(call_json("GET", f"/models/{MODEL}/samples")[1]) == n_before)

    print("== Phase D: final audit (historical files byte-identical) ==")
    f_n, f_bytes, f_tmp, f_snap = audit("final")
    changed = [k for k in pre_snap if pre_snap[k] != f_snap.get(k)]
    new_files = set(f_snap) - set(pre_snap)
    expected = {f"samples/{MODEL}/sample-{x['sample_id']}/manifest.json"
                for x in (s1, s2, t1[1], t2[1])}
    check("D1 every pre-existing file byte-identical", changed == [])
    check("D2 new files are EXACTLY the 4 sample manifests",
          new_files == expected, f"{len(new_files)} new")
    check("D3 totals 94 files / zero .tmp / samples family only",
          f_n == 94 and f_tmp == 0
          and all(k.startswith("samples/") for k in new_files)
          and (ROOT / "samples").is_dir()
          and len(list((ROOT / "samples" / MODEL).glob("sample-*"))) == 4)
    # determinism after writes: dashboard hash unchanged semantics
    dash1 = call_json("GET", f"/models/{MODEL}/dashboard")[1]
    check("D4 dashboard still deterministic and healthy",
          dash1 == call_json("GET", f"/models/{MODEL}/dashboard")[1]
          and dash1["result_hash"] == dash0["result_hash"]
          and dash1["diagnostics"] == [])

    print()
    if FAILURES:
        print(f"M15 live smoke FAILED: {len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("M15 live smoke OK: deterministic checkpoint sampling (greedy + "
          "seeded temperature, immutable sample manifests, exact storage "
          "accounting) verified live on production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
