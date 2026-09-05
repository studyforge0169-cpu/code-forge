"""Read-only regression dashboards over the immutable histories (M8 + M13).

The dashboard OBSERVES the existing M1–M12 artifacts; it never controls,
mutates or replaces anything:

    existing immutable manifests (checkpoints / evaluations / comparisons /
    gate decisions / workflow runs / model provenance / M10 suite runs /
    M12 recipe provenance on workflow runs)
            ↓
    deterministic aggregation (live recomputation, no cache, no database)
            ↓
    one JSON response

M13 adds observation only: (1) a deterministic per-model ``suite_runs``
section fed by the STORAGE-ROOT ``suite-runs/`` family (records filtered by
their persisted ``model_id`` — never copied into model directories) with
execution-bookkeeping status counts and untouched per-probe detail; and
(2) artifact-graph support for ``suite_run`` nodes (workflow → suite run →
per-probe evaluations edges, from recorded references only). No suite score,
aggregate, ranking or recipe quality number is ever computed.

Rules honoured here (tested):

  * nothing is invented: every value reported comes from a persisted
    manifest; grouping keys are derived deterministically from recorded
    fields (evaluations: the exact M4/M5/M6 probe identity; comparisons: the
    ordered state pair + probe + tolerance; gates: the recorded policy)
  * ordering never depends on filesystem traversal: artifacts are sorted by
    their persisted (created_at, id); series/groups/nodes/edges are sorted
    canonically
  * a corrupt/unreadable artifact is SKIPPED with a deterministic diagnostic
    (family + directory-derived id + issue) and valid artifacts remain
    visible — the dashboard never fabricates data and never crashes
  * the artifact graph only contains real artifacts and recorded references;
    an internal reference to a missing artifact becomes a diagnostic, never
    an invented edge
  * the response is hash-stable: ``result_hash`` covers the canonical JSON of
    the dashboard itself (minus the hash), so identical storage reproduces it
  * the engine performs no writes of any kind
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Type

from .schemas import (
    CheckpointRecord,
    ComparisonRecord,
    DashboardComparisonSeries,
    DashboardDiagnostic,
    DashboardEvaluationSeries,
    DashboardGateSeries,
    DashboardGraph,
    DashboardGraphEdge,
    DashboardGraphNode,
    DashboardSampleQuality,
    DashboardSampleQualityBySample,
    DashboardSampleQualityLatest,
    DashboardSuiteRunSummary,
    DashboardWorkflowSummary,
    EvaluationRecord,
    GateDecision,
    ModelDashboard,
    ModelRecord,
    SampleEvaluationRecord,
    SuiteRunRecord,
    WorkflowRecord,
)

log = logging.getLogger("forge.dashboards")

# Directory prefixes per artifact family (identical to the M4–M7 engines).
_FAMILIES: dict[str, tuple[str, str]] = {
    "checkpoint": ("checkpoints", ""),     # M3 dirs carry the bare checkpoint id
    "evaluation": ("evaluations", "eval-"),
    "comparison": ("comparisons", "comp-"),
    "gate_decision": ("gates", "gate-"),
    "workflow": ("workflows", "workflow-"),
}
# Families whose manifests live at the STORAGE ROOT, not under a model dir
# (M10 suite runs). Records are still model-scoped via their persisted
# ``model_id`` field; the dashboard filters on it and never copies anything.
_ROOT_FAMILIES: dict[str, tuple[str, str]] = {
    "suite_run": ("suite-runs", ""),
    # M16 sample evaluations nest per model: sample-evaluations/<model_id>/;
    # _scan resolves that extra segment below (same prefixes as M16).
    "sample_evaluation": ("sample-evaluations", "evaluation-"),
}
_MANIFEST = "manifest.json"


class DashboardEngine:
    """Deterministic, read-only aggregation over one model's artifacts."""

    def __init__(self, storage) -> None:
        self.storage = storage

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def dashboard(self, model_id: str) -> ModelDashboard:
        """Build the full per-model dashboard (never writes anything)."""
        model = self._require_model(model_id)
        diagnostics: list[DashboardDiagnostic] = []

        checkpoints = self._scan(
            model_id, "checkpoint", CheckpointRecord,
            lambda r: (r.created_at, r.checkpoint_id), diagnostics)
        evaluations = self._scan(
            model_id, "evaluation", EvaluationRecord,
            lambda r: (r.created_at, r.eval_id), diagnostics)
        comparisons = self._scan(
            model_id, "comparison", ComparisonRecord,
            lambda r: (r.created_at, r.comparison_id), diagnostics)
        gates = self._scan(
            model_id, "gate_decision", GateDecision,
            lambda r: (r.created_at, r.decision_id), diagnostics)
        workflows = self._scan(
            model_id, "workflow", WorkflowRecord,
            lambda r: (r.created_at, r.workflow_id), diagnostics)
        suite_runs = self._scan(
            model_id, "suite_run", SuiteRunRecord,
            lambda r: (r.created_at, r.suite_run_id), diagnostics)
        sample_evals = self._scan(
            model_id, "sample_evaluation", SampleEvaluationRecord,
            lambda r: (r.created_at, r.evaluation_id), diagnostics)
        self._diagnose_missing_samples(model_id, sample_evals, diagnostics)
        sample_quality = _sample_quality_section(sample_evals)

        eval_series = self._group_evaluations(evaluations)
        comp_series = self._group_comparisons(comparisons)
        gate_series = self._group_gates(gates)
        wf_counts = _counts(workflows, lambda r: r.status.value)
        sr_counts = _counts(suite_runs, lambda r: r.status.value)
        graph = self._build_graph(model, checkpoints, evaluations,
                                  comparisons, gates, workflows, suite_runs,
                                  diagnostics)
        diagnostics.sort(key=lambda d: (d.family,
                                        d.artifact_id or "", d.issue))

        dash = ModelDashboard(
            model_id=model_id,
            config_hash=model.config_hash,
            summary=_model_summary(model),
            checkpoints=checkpoints,
            training_runs=list(model.training_provenance),  # manifest order
            evaluations=eval_series,
            comparisons=comp_series,
            gate_decisions=gate_series,
            workflows=DashboardWorkflowSummary(counts=wf_counts,
                                               records=workflows),
            suite_runs=DashboardSuiteRunSummary(counts=sr_counts,
                                                records=suite_runs),
            sample_quality=sample_quality,
            artifact_graph=graph,
            diagnostics=diagnostics,
        )
        payload = dash.model_dump(mode="json", exclude={"result_hash"})
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return dash.model_copy(
            update={"result_hash": hashlib.sha256(blob).hexdigest()})

    # ------------------------------------------------------------------ #
    # Scanning with corruption resilience
    # ------------------------------------------------------------------ #

    def _scan(self, model_id: str, family: str, schema_cls: Type,
              sort_key, diagnostics: list[DashboardDiagnostic]) -> list:
        """Read one artifact family from disk.

        Directory names decide membership (same prefixes as the engines);
        unreadable manifests (missing file, invalid JSON, schema mismatch)
        are skipped and reported in ``diagnostics`` with the id derived from
        the directory name. Records are sorted by their persisted
        (created_at, id) — never by directory order.
        """
        if family in _ROOT_FAMILIES:
            subdir, prefix = _ROOT_FAMILIES[family]
            root = self.storage.root / subdir
            if family == "sample_evaluation":
                # sample-evaluations/<model_id>/evaluation-<id>/ — the
                # model segment keeps every model's records apart
                root = root / model_id
        else:
            subdir, prefix = _FAMILIES[family]
            root = self.storage.model_dir(model_id) / subdir
        if not root.exists():
            return []
        records = []
        for d in sorted(root.iterdir()):
            if not d.is_dir() or not d.name.startswith(prefix):
                continue
            artifact_id = d.name[len(prefix):]
            mpath = d / _MANIFEST
            if not mpath.exists():
                diagnostics.append(DashboardDiagnostic(
                    family=family, artifact_id=artifact_id,
                    issue="manifest.json missing"))
                continue
            try:
                rec = schema_cls(**read_json(mpath))
            except Exception as exc:  # noqa: BLE001 - one bad file never kills the view
                diagnostics.append(DashboardDiagnostic(
                    family=family, artifact_id=artifact_id,
                    issue=f"{type(exc).__name__}: {_clip(str(exc))}"))
                continue
            # root-level families hold every model's records: keep only the
            # ones this dashboard is scoped to (silent, not a diagnostic)
            if getattr(rec, "model_id", model_id) != model_id:
                continue
            records.append(rec)
        records.sort(key=sort_key)
        return records

    # ------------------------------------------------------------------ #
    # Grouping (deterministic identity keys, M4–M6 semantics preserved)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _group_evaluations(records: list[EvaluationRecord]):
        groups: dict[str, list[EvaluationRecord]] = {}
        identities: dict[str, dict[str, Any]] = {}
        for rec in records:
            identity = _eval_probe_identity(rec)
            key = _canonical_key(identity)
            identities.setdefault(key, identity)
            groups.setdefault(key, []).append(rec)
        return [DashboardEvaluationSeries(
                    identity=identities[k], key=k, count=len(groups[k]),
                    records=groups[k])
                for k in sorted(groups)]

    @staticmethod
    def _group_comparisons(records: list[ComparisonRecord]):
        groups: dict[str, list[ComparisonRecord]] = {}
        identities: dict[str, dict[str, Any]] = {}
        for rec in records:
            identity = _comparison_identity(rec)
            key = _canonical_key(identity)
            identities.setdefault(key, identity)
            groups.setdefault(key, []).append(rec)
        return [DashboardComparisonSeries(
                    identity=identities[k], key=k, count=len(groups[k]),
                    records=groups[k])
                for k in sorted(groups)]

    @staticmethod
    def _group_gates(records: list[GateDecision]):
        groups: dict[str, list[GateDecision]] = {}
        for rec in records:
            policy = rec.policy.model_dump(mode="json")
            key = hashlib.sha256(
                json.dumps(policy, sort_keys=True, default=str)
                .encode("utf-8")).hexdigest()
            groups.setdefault(key, []).append(rec)
        out = []
        for k in sorted(groups):
            recs = groups[k]
            out.append(DashboardGateSeries(
                policy_key=k, policy=recs[0].policy.model_dump(mode="json"),
                statistics=_counts(recs, lambda r: r.decision.value),
                records=recs))
        return out

    # ------------------------------------------------------------------ #
    # Artifact reference graph
    # ------------------------------------------------------------------ #

    def _build_graph(self, model: ModelRecord, checkpoints, evaluations,
                     comparisons, gates, workflows, suite_runs,
                     diagnostics: list[DashboardDiagnostic]) -> DashboardGraph:
        nodes: set[tuple[str, str]] = set()
        nodes.update(("checkpoint", r.checkpoint_id) for r in checkpoints)
        nodes.update(("evaluation", r.eval_id) for r in evaluations)
        nodes.update(("comparison", r.comparison_id) for r in comparisons)
        nodes.update(("gate_decision", r.decision_id) for r in gates)
        nodes.update(("workflow", r.workflow_id) for r in workflows)
        nodes.update(("suite_run", r.suite_run_id) for r in suite_runs)
        run_ids = [p.run_id for p in model.training_provenance]
        nodes.update(("training_run", rid) for rid in run_ids)

        edges: set[tuple[str, str, str, str, str]] = set()
        missing: list[tuple[str, str, str, str, str]] = []

        def link(src_f: str, src_id: str, dst_f: str, dst_id: str,
                 role: str) -> None:
            if not dst_id or not src_id:
                return
            if (dst_f, dst_id) in nodes:
                edges.add((src_f, src_id, dst_f, dst_id, role))
            else:
                missing.append((src_f, src_id, dst_f, dst_id, role))

        for r in checkpoints:
            link("checkpoint", r.checkpoint_id, "checkpoint",
                 r.parent_checkpoint_id, "parent")
            if r.run_id in run_ids:
                link("training_run", r.run_id, "checkpoint",
                     r.checkpoint_id, "produced")
        for r in evaluations:
            if r.state_kind.value == "checkpoint":
                link("evaluation", r.eval_id, "checkpoint",
                     r.checkpoint_id, "state")
        for r in comparisons:
            link("comparison", r.comparison_id, "evaluation",
                 r.state_a.evaluation_id, "state_a")
            link("comparison", r.comparison_id, "evaluation",
                 r.state_b.evaluation_id, "state_b")
        for r in gates:
            link("gate_decision", r.decision_id, "evaluation",
                 r.candidate.evaluation_id, "candidate")
            if r.baseline is not None:
                link("gate_decision", r.decision_id, "evaluation",
                     r.baseline.evaluation_id, "baseline")
            link("gate_decision", r.decision_id, "comparison",
                 r.comparison_id, "evidence")
            link("gate_decision", r.decision_id, "checkpoint",
                 r.suggested_checkpoint_id, "suggestion")
        for r in workflows:
            for st in r.stages:
                art = st.artifact
                if art is None:
                    continue
                dst = {"evaluation": "evaluation",
                       "comparison": "comparison",
                       "gate_decision": "gate_decision",
                       "training_report": "training_run",
                       "suite_run": "suite_run"}.get(art.kind.value)
                if dst is not None:
                    link("workflow", r.workflow_id, dst, art.artifact_id,
                         "stage_artifact")
        # M10 suite runs reference their per-probe M4 evaluations (created or
        # reused); edges exist only when the evaluation is really there.
        for r in suite_runs:
            for probe in r.results:
                if probe.evaluation_id:
                    link("suite_run", r.suite_run_id, "evaluation",
                         probe.evaluation_id, "probe")
        for p in model.training_provenance:
            link("training_run", p.run_id, "checkpoint",
                 p.final_checkpoint_id, "final_state")
            link("training_run", p.run_id, "checkpoint",
                 p.rolled_back_to, "rolled_back_to")

        for src_f, src_id, dst_f, dst_id, role in sorted(missing):
            diagnostics.append(DashboardDiagnostic(
                family=src_f, artifact_id=src_id,
                issue=f"references missing {dst_f} '{dst_id}' "
                      f"(role '{role}')"))

        nodes_sorted = sorted(
            (DashboardGraphNode(family=f, artifact_id=i) for f, i in nodes),
            key=lambda n: (n.family, n.artifact_id))
        edges_sorted = sorted(
            (DashboardGraphEdge(
                source=DashboardGraphNode(family=a, artifact_id=b),
                target=DashboardGraphNode(family=c, artifact_id=d),
                role=e)
             for a, b, c, d, e in edges),
            key=lambda ed: (ed.source.family, ed.source.artifact_id,
                            ed.target.family, ed.target.artifact_id, ed.role))
        return DashboardGraph(nodes=nodes_sorted, edges=edges_sorted)

    # ------------------------------------------------------------------ #
    # Model existence / integrity
    # ------------------------------------------------------------------ #

    def _diagnose_missing_samples(self, model_id: str, records,
                                  diagnostics: list[DashboardDiagnostic]
                                  ) -> None:
        """Sample evaluations reference their immutable M15 sample; when
        that sample directory is absent from samples/<model_id>/, surface a
        deterministic diagnostic (existing dashboard convention for
        internal references to missing artifacts). Directory-name existence
        only — sample manifests are never parsed here."""
        sdir = self.storage.root / "samples" / model_id
        if not records:
            return
        existing = set()
        if sdir.exists():
            existing = {d.name[len("sample-"):]
                        for d in sdir.iterdir()
                        if d.is_dir() and d.name.startswith("sample-")}
        for rec in records:      # records already in (created_at, id) order
            if rec.sample_id not in existing:
                diagnostics.append(DashboardDiagnostic(
                    family="sample_evaluation", artifact_id=rec.evaluation_id,
                    issue=f"references missing sample '{rec.sample_id}'"))

    def _require_model(self, model_id: str) -> ModelRecord:
        mdir = self.storage.model_dir(model_id)
        if not (mdir / _MANIFEST).exists():
            raise FileNotFoundError(f"model '{model_id}' not found")
        try:
            return self.storage.load_record(model_id)
        except Exception as exc:  # corrupt model manifest
            raise RuntimeError(
                f"model '{model_id}' manifest is corrupt: "
                f"{type(exc).__name__}: {_clip(str(exc))}") from exc


# --------------------------------------------------------------------------- #
# Module helpers (pure, deterministic)
# --------------------------------------------------------------------------- #

def _clip(text: str, limit: int = 160) -> str:
    return text[:limit]


def _counts(records, value_of) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        key = value_of(r)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _canonical_key(identity: dict[str, Any]) -> str:
    blob = json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _eval_probe_identity(rec: EvaluationRecord) -> dict[str, Any]:
    """Exact M4/M5/M6 evaluation identity (same fields the engines reuse on)."""
    cfg = rec.config or {}
    return {
        "state_kind": rec.state_kind.value,
        "checkpoint_id": rec.checkpoint_id,
        "state_hash": rec.state_hash,
        "dataset_id": rec.dataset_id,
        "dataset_version": rec.dataset_version,
        "split": rec.split.value,
        "tokenizer_id": rec.tokenizer_id,
        "max_eval_tokens": cfg.get("max_eval_tokens"),
        "max_seq_len": cfg.get("max_seq_len"),
        "batch_size": cfg.get("batch_size"),
        "seed": rec.seed,
    }


def _comparison_identity(rec: ComparisonRecord) -> dict[str, Any]:
    sa, sb = rec.state_a, rec.state_b
    return {
        "state_a": {"state_kind": sa.state_kind.value,
                    "checkpoint_id": sa.checkpoint_id,
                    "state_hash": sa.state_hash},
        "state_b": {"state_kind": sb.state_kind.value,
                    "checkpoint_id": sb.checkpoint_id,
                    "state_hash": sb.state_hash},
        "dataset_id": rec.dataset_id,
        "dataset_version": rec.dataset_version,
        "split": rec.split.value,
        "tokenizer_id": rec.tokenizer_id,
        "max_eval_tokens": rec.max_eval_tokens,
        "max_seq_len": rec.max_seq_len,
        "batch_size": rec.batch_size,
        "seed": rec.seed,
        "tolerance": rec.tolerance,
    }


def _model_summary(model: ModelRecord) -> dict[str, Any]:
    return {
        "id": model.id,
        "name": model.name,
        "architecture": model.architecture.value,
        "parameter_count": model.parameter_count,
        "dtype": model.dtype,
        "config_hash": model.config_hash,
        "state_hash": model.state_hash,
        "latest_checkpoint": model.latest_checkpoint,
        "best_checkpoint": model.best_checkpoint,
        "created_at": model.created_at,
        "updated_at": model.updated_at,
        "training_run_count": len(model.training_provenance),
    }


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _sample_quality_section(records) -> DashboardSampleQuality:
    """Deterministic M17 observability section over the model's valid M16
    sample-evaluation records (already sorted ascending by
    (created_at, evaluation_id)).

    * ``total_count`` — number of valid records
    * ``by_sample`` — per-sample summaries ordered by ``sample_id``; each
      carries the latest evaluation's recorded sample result hash + count
      + newest reference (records of one sample stay in global ascending
      order, so the last one is the maximum by (created_at, evaluation_id))
    * ``latest`` — every record newest-first, ordered by
      (created_at, evaluation_id) DESCENDING, references only

    References and counts only: metric values are never copied, aggregated,
    compared or ranked here.
    """
    by_sample: list[DashboardSampleQualityBySample] = []
    for sid in sorted({r.sample_id for r in records}):
        group = [r for r in records if r.sample_id == sid]
        newest = group[-1]
        by_sample.append(DashboardSampleQualityBySample(
            sample_id=sid,
            sample_result_hash=newest.sample_result_hash,
            evaluation_count=len(group),
            latest_evaluation_id=newest.evaluation_id,
            latest_evaluated_at=newest.created_at))
    latest = [DashboardSampleQualityLatest(
        evaluation_id=r.evaluation_id, sample_id=r.sample_id,
        created_at=r.created_at)
        for r in reversed(records)]
    return DashboardSampleQuality(total_count=len(records),
                                  by_sample=by_sample, latest=latest)
