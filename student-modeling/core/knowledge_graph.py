"""
Knowledge graph: per-student concept mastery, backed by SQLite + BKT.

Implements:
    FR8  — Maintains mastery probability per (student, skill)
    FR10 — Classifies mastery into strong/partial/weak using thresholds
    FR11 — Persistently stores per-student concept map
    FR12 — Records timestamped attempt history
    FR13 — Exposes current knowledge graph via the API
    FR16 — Provides data for learning path generation

Architecture: this module orchestrates BKT (bkt/predict.py) and the
database (db/database.py). It is the canonical entry point for any
external code that needs to read or update student mastery state.
"""

import logging
import math
import uuid
from typing import Optional
from bkt.predict import BKTPredictor, ColdStartPriorCalculator
from core.attempt_learning_outcome import AdaptiveAttemptContext
from db.database import get_connection, initialise_database
from config import (
    MASTERY_STRONG_THRESHOLD,
    MASTERY_WEAK_THRESHOLD,
)

logger = logging.getLogger(__name__)


class HistoricalEffectivePriorUnavailableError(RuntimeError):
    """Raised when a pre-migration BKT prior cannot be reconstructed exactly."""


class StaleAdaptiveAttemptError(RuntimeError):
    """Raised when mastery changed after an adaptive start snapshot."""


def _require_identifier(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def label_mastery(probability: float) -> str:
    """Map a mastery probability to a discrete label."""
    if probability >= MASTERY_STRONG_THRESHOLD:
        return "strong"
    if probability < MASTERY_WEAK_THRESHOLD:
        return "weak"
    return "partial"


class KnowledgeGraph:
    """
    Per-student concept mastery, persisted to SQLite and updated via BKT.

    Typical usage:
        kg = KnowledgeGraph()
        kg.process_session(student_id="u_42", attempts=[
            {"skill": "Percent Of", "correct": 0},
            {"skill": "Percent Of", "correct": 1},
        ])
        graph = kg.get_student_graph("u_42")
    """

    def __init__(
        self,
        predictor: Optional[BKTPredictor] = None,
        cold_start: Optional[ColdStartPriorCalculator] = None,
    ) -> None:
        """
        Args:
            predictor: BKT predictor (loaded from default location if None).
            cold_start: Cold-start prior calculator. If None, attempts to load
                        from default path. Falls back gracefully if unavailable.
        """
        initialise_database()
        self.predictor = predictor or BKTPredictor.load()

        # Cold-start is optional — system works without it
        try:
            self.cold_start = cold_start or ColdStartPriorCalculator()
        except FileNotFoundError:
            logger.warning(
                "Skill similarity matrix not found — cold-start transfer disabled. "
                "System will fall back to population priors for new (student, skill) pairs."
            )
            self.cold_start = None

        logger.info("KnowledgeGraph initialised")

    # ----- Student management -----

    def ensure_student(self, student_id: str) -> None:
        """Create a student record if one doesn't already exist."""
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO students (student_id) VALUES (?)",
                (student_id,),
            )

    # ----- Attempts -----

    def record_attempt(
        self,
        student_id: str,
        skill: str,
        correct: int,
        confidence: float = 1.0,
        signal_type: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        """Log a single attempt (with optional confidence weight) to the database."""
        if correct not in (0, 1):
            raise ValueError(f"correct must be 0 or 1, got {correct}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")

        self.ensure_student(student_id)

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO attempts (student_id, skill_name, correct, confidence, signal_type, session_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (student_id, skill, correct, confidence, signal_type, session_id),
            )

    def get_attempts(self, student_id: str, skill: str) -> list[tuple[int, float]]:
        """
        Return chronological list of (label, confidence) tuples for a (student, skill) pair.
        Format matches what BKTPredictor.predict() expects for graded observations.
        """
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT correct, confidence FROM attempts
                WHERE student_id = ? AND skill_name = ?
                ORDER BY created_at ASC, attempt_id ASC
                """,
                (student_id, skill),
            ).fetchall()
        return [(r["correct"], r["confidence"]) for r in rows]

    # ----- Stable BKT initialization -----

    def get_effective_initial_prior(
        self,
        student_id: str,
        skill: str,
    ) -> Optional[dict]:
        """Return the immutable persisted BKT prior for a student-skill."""
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT effective_initial_prior, prior_source, selected_at
                FROM bkt_initial_priors
                WHERE student_id = ? AND skill_name = ?
                """,
                (student_id, skill),
            ).fetchone()

        if row is None:
            return None

        return {
            "skill": skill,
            "effective_initial_prior": row["effective_initial_prior"],
            "prior_source": row["prior_source"],
            "selected_at": row["selected_at"],
        }

    def _select_new_effective_initial_prior(
        self,
        student_id: str,
        skill: str,
    ) -> tuple[float, str, Optional[dict]]:
        """Run the existing cold-start formula once for a new pair."""
        population_prior = float(self.predictor.params[skill]["prior"])

        if self.cold_start is None:
            return population_prior, "population", {
                "population_prior": population_prior,
                "transferred_prior": population_prior,
                "related_skills_used": [],
                "evidence": "Cold-start transfer unavailable",
            }

        student_masteries = {
            entry["skill"]: entry["mastery_probability"]
            for entry in self.get_student_graph(student_id)
            if entry["skill"] != skill
        }
        result = self.cold_start.compute_prior(
            skill=skill,
            student_masteries=student_masteries,
            population_prior=population_prior,
        )
        selected = float(result["prior"])

        if not math.isfinite(selected) or not 0.0 <= selected <= 1.0:
            raise ValueError(
                "Cold-start calculator returned a prior outside [0, 1]"
            )

        used_transfer = bool(result["used_transfer"])
        source = "transfer" if used_transfer else "population"
        details = {
            "population_prior": population_prior,
            "transferred_prior": selected,
            "related_skills_used": result["related_skills_used"],
            "evidence": result["transfer_evidence"],
        }
        return selected, source, details

    def _persist_effective_initial_prior_once(
        self,
        student_id: str,
        skill: str,
        prior: float,
        source: str,
    ) -> dict:
        """Persist once and return the winner of any concurrent insert."""
        self.ensure_student(student_id)

        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO bkt_initial_priors
                    (student_id, skill_name, effective_initial_prior, prior_source)
                VALUES (?, ?, ?, ?)
                """,
                (student_id, skill, prior, source),
            )
            row = conn.execute(
                """
                SELECT effective_initial_prior, prior_source, selected_at
                FROM bkt_initial_priors
                WHERE student_id = ? AND skill_name = ?
                """,
                (student_id, skill),
            ).fetchone()

        if row is None:
            raise RuntimeError("Failed to persist effective BKT initial prior")

        return {
            "skill": skill,
            "effective_initial_prior": row["effective_initial_prior"],
            "prior_source": row["prior_source"],
            "selected_at": row["selected_at"],
        }

    def _rebase_legacy_prior_to_population(
        self,
        student_id: str,
        skill: str,
    ) -> dict:
        """Make the existing population fallback explicit and auditable."""
        population_prior = float(self.predictor.params[skill]["prior"])
        self.ensure_student(student_id)

        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO bkt_initial_priors
                    (student_id, skill_name, effective_initial_prior, prior_source)
                VALUES (?, ?, NULL, 'legacy_unknown')
                """,
                (student_id, skill),
            )
            conn.execute(
                """
                UPDATE bkt_initial_priors
                SET effective_initial_prior = ?,
                    prior_source = 'historical_population_rebase',
                    selected_at = CURRENT_TIMESTAMP
                WHERE student_id = ? AND skill_name = ?
                  AND prior_source = 'legacy_unknown'
                """,
                (population_prior, student_id, skill),
            )
            row = conn.execute(
                """
                SELECT effective_initial_prior, prior_source, selected_at
                FROM bkt_initial_priors
                WHERE student_id = ? AND skill_name = ?
                """,
                (student_id, skill),
            ).fetchone()

        if row is None or row["effective_initial_prior"] is None:
            raise RuntimeError("Failed to resolve historical BKT prior state")

        return {
            "skill": skill,
            "effective_initial_prior": row["effective_initial_prior"],
            "prior_source": row["prior_source"],
            "selected_at": row["selected_at"],
        }

    def _effective_prior_for_update(
        self,
        student_id: str,
        skill: str,
        previous_record: Optional[dict],
    ) -> tuple[dict, Optional[dict], bool]:
        persisted = self.get_effective_initial_prior(student_id, skill)
        if (
            persisted is not None
            and persisted["prior_source"] != "legacy_unknown"
        ):
            return persisted, None, False

        if persisted is not None or previous_record is not None:
            # Compatibility policy for a row created before this migration:
            # the original transfer prior cannot be reconstructed.  Preserve
            # the old production behavior explicitly by rebasing full history
            # from the population prior, mark that provenance, and use it for
            # every future update.  Adaptive starts reject such rows before
            # this one-time non-adaptive rebase, so the discontinuity is never
            # reported as a learning outcome.
            population_prior = float(self.predictor.params[skill]["prior"])
            persisted = self._rebase_legacy_prior_to_population(
                student_id,
                skill,
            )
            compatibility_rebased = (
                persisted["prior_source"]
                == "historical_population_rebase"
            )
            if compatibility_rebased:
                logger.warning(
                    "Historical mastery for %s/%s had no recoverable "
                    "effective initial prior; explicitly rebasing full "
                    "history to the population prior %.6f. This transition "
                    "is not eligible for an adaptive learning outcome.",
                    student_id,
                    skill,
                    population_prior,
                )
            return persisted, None, compatibility_rebased

        selected, source, details = self._select_new_effective_initial_prior(
            student_id,
            skill,
        )
        persisted = self._persist_effective_initial_prior_once(
            student_id,
            skill,
            selected,
            source,
        )

        if (
            persisted["prior_source"] != source
            or not math.isclose(
                persisted["effective_initial_prior"],
                selected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            # Another writer selected the immutable value first.
            details = None

        return persisted, details, False

    def get_current_mastery_probability(
        self,
        student_id: str,
        skill: str,
        *,
        require_effective_prior: bool = False,
    ) -> Optional[float]:
        """Read authoritative current BKT mastery without persisting an update."""
        mastery = self.get_mastery(student_id, skill)
        persisted = self.get_effective_initial_prior(student_id, skill)

        prior_is_unknown = (
            persisted is None
            or persisted["effective_initial_prior"] is None
            or persisted["prior_source"] == "legacy_unknown"
        )

        if mastery is not None:
            if require_effective_prior and prior_is_unknown:
                raise HistoricalEffectivePriorUnavailableError(
                    f"Historical mastery for {student_id}/{skill} has no "
                    "recoverable effective initial prior. Run a documented "
                    "non-adaptive compatibility rebase before starting an "
                    "adaptive attempt."
                )
            return float(mastery["mastery_probability"])

        attempts = self.get_attempts(student_id, skill)
        if prior_is_unknown:
            if attempts:
                raise HistoricalEffectivePriorUnavailableError(
                    f"Stored attempts for {student_id}/{skill} predate "
                    "effective-prior persistence; the original prior cannot "
                    "be reconstructed exactly."
                )
            return None

        prior = float(persisted["effective_initial_prior"])
        if not attempts:
            return prior

        return float(
            self.predictor.predict(
                skill,
                attempts,
                initial_prior=prior,
            )
        )

    def start_attempt(
        self,
        student_id: str,
        attempt_id: str,
        skill: str,
    ) -> AdaptiveAttemptContext:
        """Capture start mastery without creating an observation or mastery row."""
        student_id = _require_identifier(student_id, "student_id")
        attempt_id = _require_identifier(attempt_id, "attempt_id")
        skill = _require_identifier(skill, "skill")

        if not self.predictor.has_skill(skill):
            raise KeyError(f"Skill '{skill}' not found in trained BKT model")

        mastery = self.get_mastery(student_id, skill)
        persisted = self.get_effective_initial_prior(student_id, skill)

        prior_is_unknown = (
            persisted is None
            or persisted["effective_initial_prior"] is None
            or persisted["prior_source"] == "legacy_unknown"
        )

        if mastery is not None and prior_is_unknown:
            raise HistoricalEffectivePriorUnavailableError(
                f"Historical mastery for {student_id}/{skill} has no "
                "recoverable effective initial prior. A non-adaptive "
                "compatibility rebase is required before adaptive use."
            )

        attempts = self.get_attempts(student_id, skill)
        if mastery is None and prior_is_unknown:
            if persisted is not None or attempts:
                raise HistoricalEffectivePriorUnavailableError(
                    f"Historical state for {student_id}/{skill} has no "
                    "recoverable effective initial prior."
                )

            selected, source, _details = (
                self._select_new_effective_initial_prior(student_id, skill)
            )
            persisted = self._persist_effective_initial_prior_once(
                student_id,
                skill,
                selected,
                source,
            )

        if mastery is not None:
            mastery_before = float(mastery["mastery_probability"])
        else:
            prior = float(persisted["effective_initial_prior"])
            mastery_before = (
                float(
                    self.predictor.predict(
                        skill,
                        attempts,
                        initial_prior=prior,
                    )
                )
                if attempts
                else prior
            )

        return AdaptiveAttemptContext(
            student_id=student_id,
            attempt_id=attempt_id,
            skill=skill,
            mastery_before=mastery_before,
        )

    def validate_attempt_context(
        self,
        context: AdaptiveAttemptContext,
    ) -> None:
        """Fail before mutation if a start snapshot is stale or unverifiable."""
        if not isinstance(context, AdaptiveAttemptContext):
            raise TypeError("context must be an AdaptiveAttemptContext")

        current = self.get_current_mastery_probability(
            context.student_id,
            context.skill,
            require_effective_prior=True,
        )
        if current is None or not math.isclose(
            current,
            context.mastery_before,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise StaleAdaptiveAttemptError(
                "Adaptive attempt mastery snapshot is stale; no evaluation "
                "evidence was persisted."
            )

    # ----- Mastery -----

    def update_mastery(self, student_id: str, skill: str) -> dict:
        """
        Recalculate mastery for (student, skill) using full attempt history,
        persist the new value, and store the previous value for regression detection.

        On first encounter, select and persist the exact effective prior once.
        Every complete-history recomputation then reuses that same value.

        Returns:
            Dict with 'probability', 'label', 'cold_start_used', and
            'cold_start_details' fields.
        """
        if not self.predictor.has_skill(skill):
            logger.warning(f"Skill '{skill}' not in BKT model — skipping update")
            return {
                "probability": 0.0,
                "label": "weak",
                "cold_start_used": False,
                "cold_start_details": None,
            }

        attempts = self.get_attempts(student_id, skill)
        if not attempts:
            logger.warning(f"No attempts recorded for {student_id}/{skill}")
            return {
                "probability": 0.0,
                "label": "weak",
                "cold_start_used": False,
                "cold_start_details": None,
            }

        previous_record = self.get_mastery(student_id, skill)
        prior_record, cold_start_details, compatibility_rebased = (
            self._effective_prior_for_update(
                student_id,
                skill,
                previous_record,
            )
        )
        effective_initial_prior = float(
            prior_record["effective_initial_prior"]
        )
        cold_start_used = prior_record["prior_source"] == "transfer"

        if cold_start_details is None and cold_start_used:
            cold_start_details = {
                "transferred_prior": effective_initial_prior,
                "source": "persisted_effective_initial_prior",
            }

        probability = self.predictor.predict(
            skill,
            attempts,
            initial_prior=effective_initial_prior,
        )
        label = label_mastery(probability)

        with get_connection() as conn:
            existing = conn.execute(
                "SELECT mastery_probability FROM mastery WHERE student_id = ? AND skill_name = ?",
                (student_id, skill),
            ).fetchone()
            previous = existing["mastery_probability"] if existing else None

            conn.execute(
                """
                INSERT INTO mastery
                    (student_id, skill_name, mastery_probability, mastery_label, previous_mastery_probability)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(student_id, skill_name) DO UPDATE SET
                    previous_mastery_probability = mastery.mastery_probability,
                    mastery_probability = excluded.mastery_probability,
                    mastery_label = excluded.mastery_label,
                    last_updated = CURRENT_TIMESTAMP
                """,
                (student_id, skill, probability, label, previous),
            )

        logger.info(
            f"Updated mastery: student={student_id} skill='{skill}' "
            f"P={probability:.3f} ({label})"
            + (" [cold-start transfer applied]" if cold_start_used else "")
        )

        return {
            "probability": probability,
            "label": label,
            "cold_start_used": cold_start_used,
            "cold_start_details": cold_start_details,
            "effective_initial_prior": effective_initial_prior,
            "effective_initial_prior_source": prior_record["prior_source"],
            "compatibility_rebased": compatibility_rebased,
        }

    def get_mastery(self, student_id: str, skill: str) -> Optional[dict]:
        """Return current stored mastery for (student, skill), or None if not present."""
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT mastery_probability, mastery_label,
                       previous_mastery_probability, last_updated
                FROM mastery
                WHERE student_id = ? AND skill_name = ?
                """,
                (student_id, skill),
            ).fetchone()

        if not row:
            return None

        return {
            "skill": skill,
            "mastery_probability": row["mastery_probability"],
            "mastery_label": row["mastery_label"],
            "previous_mastery_probability": row["previous_mastery_probability"],
            "last_updated": row["last_updated"],
        }

    def get_student_graph(self, student_id: str) -> list[dict]:
        """Return all mastery records for a student, sorted by skill name."""
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT skill_name, mastery_probability, mastery_label,
                       previous_mastery_probability, last_updated
                FROM mastery
                WHERE student_id = ?
                ORDER BY skill_name
                """,
                (student_id,),
            ).fetchall()

        return [
            {
                "skill": r["skill_name"],
                "mastery_probability": r["mastery_probability"],
                "mastery_label": r["mastery_label"],
                "previous_mastery_probability": r["previous_mastery_probability"],
                "last_updated": r["last_updated"],
            }
            for r in rows
        ]

    # ----- Session orchestration -----

    def process_session(
        self,
        student_id: str,
        signals: list[dict],
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Process a full session: record all signals, then update mastery
        for each affected skill.

        Args:
            student_id: Student identifier.
            signals: List of signal dicts. Each must contain {skill, label, confidence}
                     and optionally signal_type. Backward-compatible with the
                     older {skill, correct} format (treated as confidence=1.0).
            session_id: Optional session identifier (auto-generated if None).

        Returns:
            Summary dict with session_id, skills_updated, and updated graph.
        """
        session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        self.ensure_student(student_id)

        # Backward-compatibility shim: convert old {skill, correct} format
        normalised = []
        for s in signals:
            if "label" in s:
                normalised.append({
                    "skill": s["skill"],
                    "label": s["label"],
                    "confidence": s.get("confidence", 1.0),
                    "signal_type": s.get("signal_type"),
                })
            elif "correct" in s:
                normalised.append({
                    "skill": s["skill"],
                    "label": s["correct"],
                    "confidence": 1.0,
                    "signal_type": None,
                })
            else:
                raise ValueError(f"Signal missing label/correct: {s}")

        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions (session_id, student_id, concept_count)
                VALUES (?, ?, ?)
                """,
                (session_id, student_id, len({s["skill"] for s in normalised})),
            )

        for s in normalised:
            self.record_attempt(
                student_id=student_id,
                skill=s["skill"],
                correct=s["label"],
                confidence=s["confidence"],
                signal_type=s["signal_type"],
                session_id=session_id,
            )

        affected_skills = sorted({s["skill"] for s in normalised})
        skill_updates = []
        for skill in affected_skills:
            update_result = self.update_mastery(student_id, skill)
            skill_updates.append({"skill": skill, **update_result})

        return {
            "session_id": session_id,
            "skills_updated": skill_updates,
            "graph": self.get_student_graph(student_id),
        }

    def process_resolved_events(
        self,
        student_id: str,
        resolved_events: list,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Process Signal Resolver outputs.

        Each resolved event can create at most one BKT observation. Behavioural
        evidence is preserved in the returned summary, but does not create
        additional attempts or independently affect mastery.
        """
        session_id = session_id or f"sess_{uuid.uuid4().hex[:12]}"
        self.ensure_student(student_id)

        affected_skills = set()
        behaviour_events = []

        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sessions
                    (session_id, student_id, concept_count)
                VALUES (?, ?, ?)
                """,
                (
                    session_id,
                    student_id,
                    len({event.skill_id for event in resolved_events}),
                ),
            )

        for event in resolved_events:
            behaviour_events.append({
                "event_id": event.event_id,
                "skill": event.skill_id,
                "primary_signal": event.primary_signal.value,
                "reasoning_probability": event.behaviour.reasoning_probability,
                "uncertainty_probability": event.behaviour.uncertainty_probability,
                "clarification_probability": event.behaviour.clarification_probability,
                "repeated_misunderstanding": (
                    event.history.repeated_misunderstanding
                ),
            })

            if not event.bkt_update.should_update:
                continue

            self.record_attempt(
                student_id=student_id,
                skill=event.skill_id,
                correct=event.bkt_update.outcome,
                confidence=event.bkt_update.update_confidence,
                signal_type=event.primary_signal.value,
                session_id=session_id,
            )
            affected_skills.add(event.skill_id)

        skill_updates = []
        for skill in sorted(affected_skills):
            update_result = self.update_mastery(student_id, skill)
            skill_updates.append({"skill": skill, **update_result})

        return {
            "session_id": session_id,
            "skills_updated": skill_updates,
            "behaviour_events": behaviour_events,
            "graph": self.get_student_graph(student_id),
        }
