"""Stable policy registry and named probe suites (Milestone 9).

M9 ORGANIZES AND NAMES what M6/M4 already know — it invents no new quality
system, decision mathematics or probe identity:

    PolicyDefinition  = immutable name for one exact M6 GatePolicy
    ProbeSuite        = immutable name for a SET of exact M4 probes

  * policies live at <root>/policies/<policy_id>/manifest.json and suites at
    <root>/probe-suites/<suite_id>/manifest.json — one atomic manifest each,
    nothing else (no database, no second gate engine, no duplicated decision
    history; definitions stay separate from execution/audit events)
  * registration is immutable: the same id with identical semantic content is
    idempotently resolved to the existing definition; the same id with any
    different content raises ValueError ("... already exists ...") so the API
    maps it to 409. Nothing is ever overwritten or mutated
  * ``config_hash`` (policy) / ``probes_hash`` (suite) are sha256 over the
    canonical semantic JSON only — ids, timestamps, descriptions and paths
    are excluded, so identical semantics always reproduce the hash
  * probes are stored in canonical order and in-suite duplicates are rejected
    (a suite is a set of exact M4 probes); ordering never depends on
    filesystem traversal
  * ``SuiteProbe.to_evaluation_config(model_id, checkpoint_id)`` resolves a
    probe to an M4 EvaluationConfig whose fields pass through verbatim —
    no second incompatible probe identity exists
  * list/get never writes; registration writes only the one new manifest

The engines resolve registry references (GateRequest.policy_id,
WorkflowGateStage.policy_id) through ``PolicyEngine.resolve`` and then run the
existing M6 logic; the executed policy is embedded in the GateDecision exactly
as for inline policies (plus registry provenance ids).
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Optional

from .schemas import (
    EvaluationConfig,
    GatePolicy,
    PolicyCreateRequest,
    PolicyDefinition,
    ProbeSuite,
    ProbeSuiteCreateRequest,
    SuiteProbe,
    utcnow,
)
from .storage import atomic_write_json, read_json

POLICIES_DIR = "policies"
SUITES_DIR = "probe-suites"

log = logging.getLogger("forge.policies")


def policy_config_hash(policy: GatePolicy) -> str:
    """Deterministic hash of one GatePolicy's semantic configuration only."""
    blob = json.dumps(policy.model_dump(mode="json"), sort_keys=True,
                      default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _canonical_probe_json(probe: SuiteProbe) -> str:
    return json.dumps(probe.model_dump(mode="json"), sort_keys=True,
                      default=str)


def probes_hash(probes: list[SuiteProbe]) -> str:
    """Deterministic hash over the canonical (order-independent) probe set."""
    keys = sorted(_canonical_probe_json(p) for p in probes)
    blob = json.dumps(keys, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class PolicyEngine:
    """Immutable registry of named policies and probe suites (definitions)."""

    def __init__(self, storage) -> None:
        self.storage = storage

    # ------------------------------------------------------------------ #
    # Layout (single place where definition paths are defined)
    # ------------------------------------------------------------------ #

    def policies_root(self) -> Path:
        return self.storage.root / POLICIES_DIR

    def suites_root(self) -> Path:
        return self.storage.root / SUITES_DIR

    def _policy_dir(self, policy_id: str) -> Path:
        return self.policies_root() / policy_id

    def _suite_dir(self, suite_id: str) -> Path:
        return self.suites_root() / suite_id

    # ------------------------------------------------------------------ #
    # Policies
    # ------------------------------------------------------------------ #

    def register_policy(self, request: PolicyCreateRequest) -> PolicyDefinition:
        """Register one immutable policy; idempotent for identical content."""
        cfg_hash = policy_config_hash(request.policy)
        d = self._policy_dir(request.policy_id)
        manifest = d / "manifest.json"
        if manifest.exists():
            existing = PolicyDefinition(**read_json(manifest))
            if existing.config_hash != cfg_hash:
                raise ValueError(
                    f"a policy named '{request.policy_id}' already exists "
                    f"(config hash {existing.config_hash[:12]}…); policies are "
                    f"immutable — register a new id for different semantics")
            return existing
        record = PolicyDefinition(
            policy_id=request.policy_id,
            description=request.description,
            policy=request.policy,
            config_hash=cfg_hash,
            created_at=utcnow(),
        )
        d.mkdir(parents=True, exist_ok=False)
        atomic_write_json(manifest, record.model_dump(mode="json"))
        log.info("policy '%s' registered (config %s…)", request.policy_id,
                 cfg_hash[:12])
        return record

    def list_policies(self) -> list[PolicyDefinition]:
        root = self.policies_root()
        if not root.exists():
            return []
        records = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            mpath = d / "manifest.json"
            if mpath.exists():
                try:
                    records.append(PolicyDefinition(**read_json(mpath)))
                except Exception as exc:  # noqa: BLE001 - one bad file never hides the rest
                    log.warning("policy dir %s unreadable: %s", d.name, exc)
        records.sort(key=lambda r: (r.created_at, r.policy_id))
        return records

    def get_policy(self, policy_id: str) -> PolicyDefinition:
        mpath = self._policy_dir(policy_id) / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(f"policy '{policy_id}' not found")
        return PolicyDefinition(**read_json(mpath))

    def resolve_policy(self, model_id: str,
                       policy_id: str) -> tuple[GatePolicy, str]:
        """Resolve a registry policy for gate execution (never writes).

        Returns (policy, config_hash) after verifying the definition targets
        ``model_id``. Unreadable/corrupt definitions raise RuntimeError; a
        missing id raises FileNotFoundError — both before anything runs.
        """
        mpath = self._policy_dir(policy_id) / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(f"policy '{policy_id}' not found")
        try:
            definition = PolicyDefinition(**read_json(mpath))
        except Exception as exc:
            raise RuntimeError(
                f"policy '{policy_id}' manifest is corrupt: "
                f"{type(exc).__name__}: {exc}") from exc
        if definition.policy.model_id != model_id:
            raise ValueError(
                f"policy '{policy_id}' targets model "
                f"'{definition.policy.model_id}', gate request targets "
                f"'{model_id}'")
        return definition.policy, definition.config_hash

    # ------------------------------------------------------------------ #
    # Probe suites
    # ------------------------------------------------------------------ #

    def register_suite(self, request: ProbeSuiteCreateRequest) -> ProbeSuite:
        """Register one immutable suite; idempotent for identical contents.

        Probes are canonicalized (sorted) before hashing and persistence:
        order carries no evaluation meaning and must never influence the
        suite's identity or output ordering.
        """
        probes = sorted(request.probes,
                        key=lambda p: _canonical_probe_json(p))
        phash = probes_hash(probes)
        d = self._suite_dir(request.suite_id)
        manifest = d / "manifest.json"
        if manifest.exists():
            existing = ProbeSuite(**read_json(manifest))
            if existing.probes_hash != phash:
                raise ValueError(
                    f"a probe suite named '{request.suite_id}' already exists "
                    f"(probes hash {existing.probes_hash[:12]}…); suites are "
                    f"immutable — register a new id for different contents")
            return existing
        record = ProbeSuite(
            suite_id=request.suite_id,
            description=request.description,
            probes=probes,
            probes_hash=phash,
            created_at=utcnow(),
        )
        d.mkdir(parents=True, exist_ok=False)
        atomic_write_json(manifest, record.model_dump(mode="json"))
        log.info("probe suite '%s' registered (%d probes, hash %s…)",
                 request.suite_id, len(probes), phash[:12])
        return record

    def list_suites(self) -> list[ProbeSuite]:
        root = self.suites_root()
        if not root.exists():
            return []
        records = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            mpath = d / "manifest.json"
            if mpath.exists():
                try:
                    records.append(ProbeSuite(**read_json(mpath)))
                except Exception as exc:  # noqa: BLE001
                    log.warning("suite dir %s unreadable: %s", d.name, exc)
        records.sort(key=lambda r: (r.created_at, r.suite_id))
        return records

    def get_suite(self, suite_id: str) -> ProbeSuite:
        mpath = self._suite_dir(suite_id) / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(f"probe suite '{suite_id}' not found")
        return ProbeSuite(**read_json(mpath))

    # ------------------------------------------------------------------ #
    # Resolution helpers shared with the gates/workflows engines
    # ------------------------------------------------------------------ #

    def gate_policy_for(self, model_id: str,
                        inline: Optional[GatePolicy],
                        policy_id: Optional[str],
                        ) -> tuple[GatePolicy, Optional[str], Optional[str]]:
        """One resolution point for both gate sources (never writes).

        Returns (policy, policy_id, config_hash): inline policies keep their
        id/hash as None (historical M6 behaviour unchanged); registry ids are
        resolved and verified against ``model_id``.
        """
        if policy_id is not None:
            policy, cfg_hash = self.resolve_policy(model_id, policy_id)
            return policy, policy_id, cfg_hash
        return inline, None, None  # type: ignore[return-value]

    def evaluation_config_for(self, probe: SuiteProbe, model_id: str,
                              checkpoint_id: Optional[str] = None
                              ) -> EvaluationConfig:
        """Resolve one suite probe to an exact M4 EvaluationConfig."""
        return probe.to_evaluation_config(model_id, checkpoint_id)
