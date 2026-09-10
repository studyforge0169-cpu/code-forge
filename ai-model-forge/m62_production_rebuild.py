"""M62 production-root REBUILD (evidence script, executed exactly once).

CONTEXT (2026-09-10): the sandbox running this repository was
re-provisioned between M61 and M62. The Git repository — including all
61 milestones of history — was fully recovered from the remote
(HEAD 229f860, worktree byte-identical), but the PRODUCTION DATA
DIRECTORY (/home/user/ai-model-forge-data) was lost: it deliberately
lived OUTSIDE the repository and was never committed. Its exact bytes
(model ids, timestamps, random record ids) are not reproducible.

This script reconstructs a NEW production root through the PUBLIC
engine facade (the exact code paths behind the HTTP API), mirroring
the documented SHAPE of the original production trajectory:
a dataset + tokenizer, a small transformer, several training runs
building an improvement history, a deliberate best!=latest drift, the
canonical M58 loop recipe (TRAIN resume_from_best -> EVALUATE best ->
GATE best -> PUBLISH best, repetitions=2) closing the drift through
M60, and one M61 explicit deletion of the superseded bootstrap winner.
Every number it prints is a fact about the NEW root (m62_pre.sha256
captures it byte-exactly before the M62 read-only smoke).
"""
from __future__ import annotations

import json
import os
import random
import shutil

ROOT = "/home/user/ai-model-forge-data"
os.environ["FORGE_ROOT"] = ROOT
os.environ["FORGE_LOG_LEVEL"] = "WARNING"
shutil.rmtree(ROOT, ignore_errors=True)

from app.engine import ModelForge  # noqa: E402
from app.recipes import RecipeEngine  # noqa: E402
from app.schemas import (  # noqa: E402
    EvalStateKind, GatePolicy, StageStateRef, StageType, TokenizerConfig,
    TrainingConfig, TransformerConfig, ModelCreateRequest,
    WorkflowEvaluationStage, WorkflowGateStage, WorkflowPublishStage,
    WorkflowPlan, WorkflowRecipeCreateRequest, WorkflowStage,
    EvaluationConfig)

rng = random.Random(11)
HEADS = ("river mountain cloud forest desert ocean valley island meadow "
         "canyon").split()
TAILS = ("flows stands gleams rises falls drifts looms shines hides "
         "waits").split()
body = "\n\n".join(
    f"{rng.choice(HEADS)} is {rng.choice(TAILS)} near "
    f"{rng.choice(HEADS)} with number {i}" for i in range(260))

forge = ModelForge(root=ROOT)
up = forge.upload_dataset([("corpus.txt", (body + "\n").encode())],
                          name="live-corpus")
ds = up["dataset_id"]
tok = forge.train_tokenizer(TokenizerConfig(name="live-tok",
                                            vocab_size=600),
                            dataset_id=ds)
forge.tokenize_dataset(ds, tok.id)
mid = forge.create_model(ModelCreateRequest(config=TransformerConfig(
    name="live-model", vocab_size=640, context_length=64, hidden_size=64,
    n_layers=2, n_heads=4, n_kv_heads=2, intermediate_size=128, seed=1),
    description="production model (M62 rebuild)"))[0].id
print("model:", mid)


def train(**kw):
    base = dict(method="continued_pretraining", model_id=mid, dataset_id=ds,
                tokenizer_id=tok.id, learning_rate=3e-3, batch_size=8,
                max_seq_len=32, eval_every_steps=4, keep_best=False, seed=1)
    base.update(kw)
    return forge.run_training(TrainingConfig(**base))


def state():
    m = forge.get_model(mid)
    sel = forge.select_best_checkpoint(mid)
    return (f"ckpts={len(forge.list_checkpoints(mid))} "
            f"best={sel.checkpoint.checkpoint_id}"
            f"@{sel.checkpoint.validation_loss:.6f} "
            f"latest={m.latest_checkpoint} "
            f"drift={sel.checkpoint.checkpoint_id != m.latest_checkpoint}")


# --- bootstrap improvement history (plain continuation runs) -----------
train(steps=16, seed=1)
print("run1:", state())
train(steps=12, seed=2)
print("run2:", state())
train(steps=8, seed=3)
print("run3:", state())
# a deliberate non-improving run moves `latest` WITHOUT moving `best`
# -> the real pre-M60 drift shape (best != published)
train(steps=4, seed=99)
print("run4 (drift):", state())

# --- the canonical loop, as ONE registered recipe, run twice (M58) -----
recipes = RecipeEngine(forge.storage, workflows=forge.workflows)
loop = [WorkflowStage(
    stage_id="tr", type=StageType.TRAIN,
    training=TrainingConfig(
        method="continued_pretraining", model_id=mid, dataset_id=ds,
        tokenizer_id=tok.id, learning_rate=3e-3, batch_size=8,
        max_seq_len=32, eval_every_steps=2, seed=7, steps=4,
        resume_from_best=True)),
    WorkflowStage(
        stage_id="ev", type=StageType.EVALUATE,
        evaluation=WorkflowEvaluationStage(
            config=EvaluationConfig(
                model_id=mid, dataset_id=ds, tokenizer_id=tok.id,
                split="validation", batch_size=8, max_seq_len=32),
            checkpoint_from_best=True)),
    WorkflowStage(
        stage_id="gate", type=StageType.GATE,
        gate=WorkflowGateStage(
            policy=GatePolicy(
                name="live-gate", model_id=mid, dataset_id=ds,
                tokenizer_id=tok.id, split="validation", batch_size=8,
                max_seq_len=32, seed=2, baseline_type="checkpoint",
                baseline_from_best=True, tolerance=1.0),
            candidate=StageStateRef(state_kind=EvalStateKind.BEST))),
    WorkflowStage(
        stage_id="pub", type=StageType.PUBLISH,
        publish=WorkflowPublishStage(publish_from_best=True))]
recipes.register(WorkflowRecipeCreateRequest(
    recipe_id="m62-live-loop", description="canonical improvement loop "
    "(M55+M56+M57+M60 declarations, run twice per M58)", stages=loop))
records = recipes.run_repeated("m62-live-loop", mid, 2)
print("loop:", [r.status.value for r in records], "|", state())

# --- the standalone M60 acceptance: ONE inline PUBLISH(best) workflow
# closing the drift the loop's per-plan pinning left behind (the exact
# shape of the original production's M60 smoke) -------------------------
forge.run_workflow(WorkflowPlan(
    name="m62-live-publish-best", model_id=mid,
    description="standalone M60 acceptance: PUBLISH(best)",
    stages=[WorkflowStage(stage_id="publish_best", type=StageType.PUBLISH,
                          publish=WorkflowPublishStage(
                              publish_from_best=True))]))
print("publish:", state())

# --- M61: one explicit verified deletion of the superseded first
# winner (mirroring the original production's M61 smoke) ---------------
hist = forge.best_checkpoint_history(mid)
first_winner = hist.entries[0].checkpoint_id
if first_winner != forge.select_best_checkpoint(mid).checkpoint.checkpoint_id \
        and forge.checkpoint_blockers(mid, first_winner) == []:
    res = forge.delete_checkpoint(mid, first_winner)
    print("m61 deletion:", first_winner, res.files_removed,
          res.bytes_reclaimed)
else:
    print("m61 deletion: first winner not deletable — skipped (honest)")

# --- final certified facts ----------------------------------------------
m = forge.get_model(mid)
sel = forge.select_best_checkpoint(mid)
ov = forge.checkpoint_retention_overview(mid)
summary = {
    "model_id": mid,
    "dataset_id": ds, "tokenizer_id": tok.id,
    "checkpoints": len(forge.list_checkpoints(mid)),
    "best": sel.checkpoint.checkpoint_id,
    "best_loss": sel.checkpoint.validation_loss,
    "published": m.latest_checkpoint,
    "best_equals_published":
        sel.checkpoint.checkpoint_id == m.latest_checkpoint,
    "history_entries": len(forge.best_checkpoint_history(mid).entries),
    "runs": len(m.training_provenance),
    "workflows": len(forge.list_workflows(mid)),
    "evaluations": len(forge.list_evaluations(mid)),
    "comparisons": len(forge.list_comparisons(mid)),
    "gates": len(forge.list_gate_decisions(mid)),
    "recipes": len(recipes.list()),
    "retention": {
        "total": ov.total_checkpoints,
        "deletable": ov.deletable_checkpoints,
        "protected": ov.protected_checkpoints,
        "total_bytes": ov.total_checkpoint_bytes,
        "reclaimable_bytes": ov.reclaimable_checkpoint_bytes,
    },
}
print("FINAL STATE:")
print(json.dumps(summary, indent=2))
