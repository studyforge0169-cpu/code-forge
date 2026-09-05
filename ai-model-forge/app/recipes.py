"""Named workflow recipes: immutable, reusable M7 pipeline definitions
(M12 + M14 recipe composition).

A recipe is INERT DATA — a user-chosen id plus the exact M7 ``WorkflowStage``
list — registered once and executed only through an explicit run request that
binds ONE model. M14 lets a registered recipe REFERENCE other registered
recipes via a ``recipe`` stage; the composition is expanded deterministically
into ONE normal WorkflowPlan executed by ONE WorkflowEngine run:

    API → Forge facade → RecipeEngine (registry + composition resolver)
                              ↓ (expand → bind model → ONE WorkflowPlan)
                          WorkflowEngine   ← the SOLE workflow executor
                              ├── train / evaluate / compare / gate
                              └── suite_run → SuiteRunEngine → M4

  * recipes live at <root>/workflow-recipes/<recipe_id>/manifest.json — one
    atomic manifest each, following the M9 immutable-definition conventions
    (policies / probe suites); nothing else is ever written by registration
  * registration reuses the shared M7 plan-structure validation
    (``validate_plan_stages``) so recipe rules can never drift from inline
    plans; the model-equality half of plan validation runs when the recipe is
    re-validated as a real ``WorkflowPlan`` at run time against the bound
    model
  * registration is immutable: the same recipe_id with identical semantic
    contents is idempotently resolved to the existing definition; the same id
    with any different content raises ValueError ("... already exists ...")
    so the API maps it to 409. Nothing is ever overwritten or mutated
  * ``config_hash`` for plain recipes is sha256 over the canonical JSON of
    the ORDERED stage list only (exact M12 semantics — recipe ids,
    descriptions, timestamps and paths are excluded, so identical semantics
    always reproduce the hash). For COMPOSITE recipes (declared stages with
    ``recipe`` references) the hash is sha256 over the declared stages PLUS
    the referenced recipes' ids and config hashes (in declared order): the
    referenced recipes are immutable, so their hashes pin the composition's
    transitive semantics deterministically
  * composition (M14): a ``recipe`` stage names a recipe that must ALREADY be
    registered (definitions compose definitions — unlike M12's suites/
    checkpoints, which resolve at run time). Registration validates, in
    order: direct references exist → no cycles (transitive, over the
    immutable registry) → composition depth ≤ 32 → deterministic expansion
    → the SHARED M7 validator over the fully expanded stage list (recipe
    stages must have disappeared) → composite hash → immutable persistence.
    No manifest is written when any step fails
  * expansion is deterministic static splicing with stage-id qualification:
    every stage of a referenced recipe keeps its declared order and receives
    the qualified id ``<call-stage path>.<stage id>``, where the call-stage
    path is the dot-joined chain of the recipe-call stage ids that led to it
    (top-level recipe stages stay bare). ALL internal references of the
    referenced recipe — ``from_stage`` state pointers and gate
    ``on_pass``/``on_fail`` branches to its own stages — are rewritten to
    the qualified ids, so recipe-internal references remain internal after
    expansion. References that would cross a recipe boundary (e.g. a branch
    targeting a recipe-call stage, which vanishes during expansion) are
    rejected by the shared validator over the expanded list
  * the run record of a composite invocation carries the full expansion
    provenance (additive ``composition`` list of recipe_id + config_hash in
    deterministic depth-first expansion order) while ``recipe_id`` /
    ``recipe_hash`` stay the top-level recipe the caller invoked — lineage
    ownership and composition provenance are different concepts
  * ``run`` resolves the definition, expands it (identity for plain
    recipes), verifies the explicitly supplied model exists, converts the
    stages into an existing ``WorkflowPlan`` (name = recipe id, full plan
    validation incl. embedded-config model agreement) and delegates to the
    existing ``WorkflowEngine``. RecipeEngine never executes a stage itself
    and never repeats or schedules anything
  * list/get never writes; runs write only through WorkflowEngine's normal
    per-run manifest (plus the M10/M4 evidence that is genuinely missing)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from .schemas import (
    StageType,
    WorkflowPlan,
    WorkflowRecipe,
    WorkflowRecipeCreateRequest,
    WorkflowRecipeRef,
    WorkflowRecord,
    validate_plan_stages,
    utcnow,
)
from .storage import atomic_write_json, read_json
from .workflows import WorkflowEngine

RECIPES_DIR = "workflow-recipes"
RECIPE_MANIFEST = "manifest.json"
MAX_COMPOSITION_DEPTH = 32      # M14: longest recipe-reference chain allowed
_STAGE_QUALIFIER = "."          # M14: deterministic stage-id qualification

log = __import__("logging").getLogger("forge.recipes")


def recipe_config_hash(stages: list) -> str:
    """Deterministic hash of one recipe's ordered semantic stage content.

    sha256 over the canonical JSON of the stage list — includes every stage
    configuration, reference, state specification, suite/policy id and branch;
    excludes recipe id, description, timestamps and paths. Order is semantic
    (workflows execute in order) so stages are hashed as ordered, never
    sorted. Identical semantics always reproduce the hash.

    The canonical stage serialization deliberately EXCLUDES the nullable
    ``recipe`` payload field (M14): a recipe-reference stage's semantic
    content is its referenced recipe_id, which composite hashes carry through
    the ``composition`` pairs instead — keeping the plain-recipe hash
    byte-identical to the M12 serialization that predates the field, so
    identical definitions registered before and after M14 reproduce the same
    config hash.

    This is the exact M12 hash for PLAIN recipes; the RecipeEngine uses a
    composite payload (declared stages + referenced recipe ids/config hashes)
    for recipes whose stages contain ``recipe`` references.
    """
    blob = json.dumps(
        [s.model_dump(mode="json", exclude={"recipe"}) for s in stages],
        sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _recipe_refs(stages: list) -> list:
    """Direct recipe references of a declared stage list, in stage order:
    (referenced recipe_id, the call stage carrying it)."""
    return [(st.recipe.recipe_id, st) for st in stages
            if st.type == StageType.RECIPE and st.recipe is not None]


class RecipeEngine:
    """Immutable registry of named workflow recipes + thin run resolver.

    Registration and read access mirror the M9 definition registries
    (policies/probe suites). ``run`` is deliberately thin: resolve → verify
    model → bind into a WorkflowPlan → delegate to the existing
    WorkflowEngine (the single workflow executor).
    """

    def __init__(self, storage, workflows: Optional[WorkflowEngine] = None):
        self.storage = storage
        self.workflows = workflows if workflows is not None \
            else WorkflowEngine(storage)

    # ------------------------------------------------------------------ #
    # Layout (single place where recipe definition paths live)
    # ------------------------------------------------------------------ #

    def recipes_root(self) -> Path:
        return self.storage.root / RECIPES_DIR

    def _recipe_dir(self, recipe_id: str) -> Path:
        return self.recipes_root() / recipe_id

    def _manifest(self, recipe_id: str) -> Path:
        return self._recipe_dir(recipe_id) / RECIPE_MANIFEST

    # ------------------------------------------------------------------ #
    # Registration / read (immutable definitions; never mutate)
    # ------------------------------------------------------------------ #

    def register(self, request: WorkflowRecipeCreateRequest) -> WorkflowRecipe:
        """Register one immutable recipe; idempotent for identical content.

        Plain recipes follow M12 exactly. COMPOSITE recipes (M14 — declared
        stages containing ``recipe`` references) are validated engine-side in
        a fixed order BEFORE any hash/persistence work: direct references
        must exist in the registry (definitions compose definitions) → no
        cycles (transitive over the immutable registry) → composition depth
        <= MAX_COMPOSITION_DEPTH → deterministic expansion → the SHARED M7
        validator over the fully expanded stage list (recipe stages must be
        gone) → composite config hash (declared stages + referenced ids and
        config hashes). No manifest is written when any step fails; an
        identical existing definition is returned as-is (no write), any
        different content under the same id raises ValueError -> 409.
        """
        refs = _recipe_refs(request.stages)
        composition: Optional[list[WorkflowRecipeRef]] = None
        if refs:
            # 1) direct references must already exist (each definition pins
            #    the config hash of its dependency at registration time)
            dep_hashes: list[str] = []
            for rid, _call in refs:
                try:
                    dep = self.get(rid)
                except FileNotFoundError:
                    raise ValueError(
                        f"recipe '{request.recipe_id}' references unknown "
                        f"workflow recipe '{rid}' — the referenced recipe "
                        f"must be registered first") from None
                dep_hashes.append(dep.config_hash)
            # 2) cycles: the candidate must not be reachable from its own
            #    references (transitive walk over the immutable registry)
            self._check_no_cycle(request.recipe_id, refs)
            # 3) depth: longest chain of recipes must not exceed the limit
            depth = 1 + max(self._depth(rid) for rid, _c in refs)
            if depth > MAX_COMPOSITION_DEPTH:
                raise ValueError(
                    f"recipe '{request.recipe_id}' composition depth {depth} "
                    f"exceeds the maximum of {MAX_COMPOSITION_DEPTH} recipes")
            # 4) deterministic expansion + shared M7 validation of the FULL
            #    expanded list (recipe stages must have disappeared)
            trace: list[WorkflowRecipeRef] = []
            expanded = self._expand_definition(
                request.recipe_id, request.stages, trace=trace)
            validate_plan_stages(expanded, model_id=None)
            # 5) composite hash: declared stages + referenced ids/hashes —
            #    the referenced recipes are immutable, so this pins the
            #    transitive semantics of the composition deterministically
            cfg_hash = self._composite_hash(request.stages, refs, dep_hashes)
            composition = [WorkflowRecipeRef(recipe_id=rid, config_hash=h)
                           for (rid, _c), h in zip(refs, dep_hashes)]
            assert trace  # every direct ref was expanded (validated above)
        else:
            cfg_hash = recipe_config_hash(request.stages)
        manifest = self._manifest(request.recipe_id)
        if manifest.exists():
            existing = WorkflowRecipe(**read_json(manifest))
            if existing.config_hash != cfg_hash:
                raise ValueError(
                    f"a workflow recipe named '{request.recipe_id}' already "
                    f"exists (config hash {existing.config_hash[:12]}…); "
                    f"recipes are immutable — register a new id for "
                    f"different semantics")
            return existing
        record = WorkflowRecipe(
            recipe_id=request.recipe_id,
            description=request.description,
            stages=request.stages,
            composition=composition,
            config_hash=cfg_hash,
            created_at=utcnow(),
        )
        d = self._recipe_dir(request.recipe_id)
        d.mkdir(parents=True, exist_ok=False)
        atomic_write_json(manifest, record.model_dump(mode="json"))
        log.info("workflow recipe '%s' registered (config %s…, %d stage(s))",
                 request.recipe_id, cfg_hash[:12], len(request.stages))
        return record

    # ------------------------------------------------------------------ #
    # Composition validation (M14; definition-level, before any write)
    # ------------------------------------------------------------------ #

    def _check_no_cycle(self, candidate_id: str, refs: list) -> None:
        """Reject a candidate whose transitive references reach itself.

        Recipes are immutable and the registry is append-only, so through the
        public API a cycle can only ever be closed by a reference to an
        already-registered recipe that (transitively) references the
        candidate — this walk detects exactly that, deterministically, and
        also guards registry states manipulated outside the API.
        """
        seen: set[str] = set()
        stack = [rid for rid, _c in refs]
        while stack:
            rid = stack.pop()
            if rid == candidate_id:
                raise ValueError(
                    f"workflow recipe composition cycle detected: recipe "
                    f"'{candidate_id}' is reachable from itself")
            if rid in seen:
                continue
            seen.add(rid)
            definition = self.get(rid)          # exists (checked above)
            stack.extend(r for r, _c in _recipe_refs(definition.stages))

    def _depth(self, recipe_id: str, memo: Optional[dict] = None) -> int:
        """Longest recipe-reference chain rooted at one registered recipe."""
        if memo is None:
            memo = {}
        if recipe_id in memo:
            return memo[recipe_id]
        definition = self.get(recipe_id)
        refs = _recipe_refs(definition.stages)
        if not refs:
            memo[recipe_id] = 1
            return 1
        memo[recipe_id] = 1 + max(self._depth(rid, memo) for rid, _c in refs)
        return memo[recipe_id]

    @staticmethod
    def _composite_hash(stages: list, refs: list,
                        dep_hashes: list[str]) -> str:
        """Deterministic config hash of a composite recipe.

        sha256 over the canonical JSON of the DECLARED ordered stages plus
        the direct references as (recipe_id, config_hash) pairs in declared
        order (stage serialization excludes the ``recipe`` payload field —
        reference content lives in the pairs). Referenced config hashes are
        themselves deterministic over the referenced recipes' full semantic
        content (recursively for nested compositions), so this pins the whole
        composition without flattening it; the composite's own
        recipe_id/description/timestamps/paths are excluded, and plain
        recipes keep the exact M12 hash.
        """
        payload = {
            "stages": [st.model_dump(mode="json", exclude={"recipe"})
                       for st in stages],
            "composition": [[rid, h] for (rid, _c), h in zip(refs,
                                                             dep_hashes)],
        }
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def list(self) -> list[WorkflowRecipe]:
        root = self.recipes_root()
        if not root.exists():
            return []
        records = []
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            mpath = d / RECIPE_MANIFEST
            if mpath.exists():
                try:
                    records.append(WorkflowRecipe(**read_json(mpath)))
                except Exception as exc:  # noqa: BLE001 - one bad file never hides the rest
                    log.warning("recipe dir %s unreadable: %s", d.name, exc)
        records.sort(key=lambda r: (r.created_at, r.recipe_id))
        return records

    def get(self, recipe_id: str) -> WorkflowRecipe:
        manifest = self._manifest(recipe_id)
        if not manifest.exists():
            raise FileNotFoundError(
                f"workflow recipe '{recipe_id}' not found")
        return WorkflowRecipe(**read_json(manifest))

    # ------------------------------------------------------------------ #
    # Lineage (read-only, M13)
    # ------------------------------------------------------------------ #

    def runs(self, recipe_id: str) -> list[WorkflowRecord]:
        """Cross-model lineage of ONE recipe's workflow runs (read-only).

        Unknown recipe -> FileNotFoundError (404); an existing recipe with no
        runs -> []. Every persisted workflow run whose ``recipe_id`` matches
        is returned with its own recorded ``model_id`` (a recipe may be
        bound to different models over time, so the view is recipe-scoped,
        never model-scoped). Scans the existing workflow manifests live
        through the WorkflowEngine's own read path — no index, no cache, no
        new storage, no writes. Ordering: (created_at, workflow_id).
        """
        self.get(recipe_id)                     # verify the recipe exists
        out: list[WorkflowRecord] = []
        for model_id in sorted(self.storage.model_ids()):
            try:
                for rec in self.workflows.list_workflows(model_id):
                    if rec.recipe_id == recipe_id:
                        out.append(rec)
            except FileNotFoundError:  # manifest-less model dir: not a model
                continue
        out.sort(key=lambda r: (r.created_at, r.workflow_id))
        return out

    # ------------------------------------------------------------------ #
    # Execution (thin binding layer over the existing WorkflowEngine)
    # ------------------------------------------------------------------ #

    def run(self, recipe_id: str, model_id: str) -> WorkflowRecord:
        """Execute one registered recipe against ONE explicitly named model.

        Composite recipes (M14) are expanded FIRST — deterministically, into
        ONE normal stage list — and plain recipes are their own identity, so
        every recipe ends up as exactly one WorkflowPlan executed by exactly
        one WorkflowEngine.run. Nothing is persisted on unknown recipe/model
        (FileNotFoundError -> 404) or on a binding mismatch (ValueError ->
        422 raised while the expanded plan is re-validated as a
        WorkflowPlan). Runtime stage failures go through the existing M7
        failure path exactly like inline plans: the run manifest persists
        with status ``failed`` (plus recipe provenance) and the stage
        exception is re-raised for API mapping.
        """
        definition = self.get(recipe_id)          # FileNotFoundError -> 404
        try:
            self.storage.load_record(model_id)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"model '{model_id}' not found") from None

        # Expand (identity for plain recipes) and record the deterministic
        # expansion trace as composition provenance on the run record.
        trace: list[WorkflowRecipeRef] = []
        stages = self._expand_definition(recipe_id, definition.stages,
                                         trace=trace)
        composition = trace or None

        # Bind: the model is the only runtime binding. The EXPANDED stage
        # list becomes a real WorkflowPlan (name = recipe id) and passes the
        # FULL existing plan validation — structural rules again, plus every
        # embedded config must agree with the bound model (this is what
        # rejects a run whose bound model contradicts a model pinned inside
        # a referenced recipe — nothing is ever silently rewritten).
        plan = WorkflowPlan(name=definition.recipe_id, model_id=model_id,
                            stages=stages)
        return self.workflows.run(plan, recipe_id=definition.recipe_id,
                                  recipe_hash=definition.config_hash,
                                  composition=composition)

    # ------------------------------------------------------------------ #
    # Deterministic composition expansion (M14)
    # ------------------------------------------------------------------ #

    def _expand_definition(self, recipe_id: str, stages: list,
                           prefix: str = "",
                           trace: Optional[list[WorkflowRecipeRef]] = None,
                           _path: Optional[list] = None) -> list:
        """Expand one recipe's declared stages into a flat, fully qualified
        stage list (identity when the recipe references nothing).

        Deterministic static splicing:

          * a stage of the top-level recipe keeps its declared id; a stage of
            a referenced recipe receives ``<call-path>.<stage id>`` where
            ``call-path`` is the dot-joined chain of the recipe-call stage
            ids that led to it (each call stage id is unique within its own
            recipe, so every expansion path is unique — two calls to the same
            recipe at different positions produce distinct qualified ids)
          * every INTERNAL reference of a referenced recipe (``from_stage``
            state pointers; gate ``on_pass``/``on_fail`` branches to its own
            non-call stages) is rewritten to the qualified id, so
            recipe-internal references stay internal after expansion
          * a branch that targets a recipe-call stage is left untouched and
            the shared validator then rejects it over the expanded list (the
            call stage vanished — branching into a callee is not a defined
            semantic)
          * when ``trace`` is given, every expanded callee is appended as
            (recipe_id, config_hash) in depth-first expansion order — the
            run-record composition provenance

        Guards (defense in depth for registries manipulated outside the API):
        a recipe path longer than MAX_COMPOSITION_DEPTH or a reference cycle
        is rejected before recursion can blow up.
        """
        if _path is None:
            _path = []
        if len(_path) >= MAX_COMPOSITION_DEPTH:
            raise ValueError(
                f"recipe composition depth exceeds the maximum of "
                f"{MAX_COMPOSITION_DEPTH} recipes while expanding "
                f"'{recipe_id}'")
        out: list = []
        call_ids = {st.stage_id for st in stages
                    if st.type == StageType.RECIPE}
        for st in stages:
            if st.type != StageType.RECIPE:
                out.append(self._qualify_stage(st, prefix, call_ids))
                continue
            rid = st.recipe.recipe_id
            if rid in _path:
                raise ValueError(
                    f"workflow recipe composition cycle detected at run "
                    f"time while expanding recipe '{rid}'")
            try:
                callee = self.get(rid)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"recipe '{recipe_id}' references workflow recipe "
                    f"'{rid}' which is no longer registered") from None
            if trace is not None:
                trace.append(WorkflowRecipeRef(
                    recipe_id=callee.recipe_id,
                    config_hash=callee.config_hash))
            child_prefix = prefix + st.stage_id + _STAGE_QUALIFIER
            out.extend(self._expand_definition(
                rid, callee.stages, prefix=child_prefix, trace=trace,
                _path=_path + [rid]))
        return out

    @staticmethod
    def _qualify_stage(stage, prefix: str, call_ids: set) -> object:
        """Prefix one stage id and rewrite its internal references when the
        stage comes from a referenced recipe (identity for prefix == "")."""
        if not prefix:
            return stage
        qid = prefix + stage.stage_id
        if len(qid) > 64:
            raise ValueError(
                f"qualified stage id '{qid}' exceeds 64 characters — use "
                f"shorter stage ids in referenced recipes")

        def q_ref(value):
            return None if value is None else prefix + value

        def q_state(ref):
            if ref is None or not ref.from_stage:
                return ref
            return ref.model_copy(update={"from_stage": q_ref(ref.from_stage)})

        updates = {}
        if stage.evaluation is not None:
            ev = stage.evaluation
            if ev.checkpoint_from_stage:
                updates["evaluation"] = ev.model_copy(update={
                    "checkpoint_from_stage": q_ref(ev.checkpoint_from_stage)})
        if stage.comparison is not None:
            cmp_ = stage.comparison
            sa = q_state(cmp_.state_a)
            sb = q_state(cmp_.state_b)
            if sa is not cmp_.state_a or sb is not cmp_.state_b:
                updates["comparison"] = cmp_.model_copy(
                    update={"state_a": sa, "state_b": sb})
        if stage.gate is not None:
            gate = stage.gate
            cand = q_state(gate.candidate)
            if cand is not gate.candidate:
                updates["gate"] = gate.model_copy(update={"candidate": cand})
        if stage.suite_run is not None:
            sr = stage.suite_run
            state = q_state(sr.state)
            if state is not sr.state:
                updates["suite_run"] = sr.model_copy(update={"state": state})
        # gate branches: rewrite targets that name THIS recipe's own non-call
        # stages; targets naming a recipe-call stage are left untouched and
        # are rejected by the shared validator on the expanded list
        for field in ("on_pass", "on_fail"):
            target = getattr(stage, field)
            if target is not None and target not in call_ids:
                updates[field] = q_ref(target)
        return stage.model_copy(update={"stage_id": qid, **updates})
