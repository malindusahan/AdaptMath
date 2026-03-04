import {
  ArrowRight,
  Brain,
  CheckCircle2,
  Database,
  Gauge,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import type {
  BKTCalculationTrace,
  BKTObservationTrace,
  BKTUpdateTrace,
} from "../types/tutor";

interface BKTUpdateTraceViewProps {
  trace: BKTUpdateTrace | null | undefined;
}

function number(value: number | null | undefined, digits = 4): string {
  return typeof value === "number" ? value.toFixed(digits) : "Pending";
}

function percent(value: number | null | undefined): string {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "Pending";
}

function label(value: string | null | undefined): string {
  return value ? value.replaceAll("_", " ") : "Not recorded";
}

function outcomeLabel(observation: BKTObservationTrace): string {
  if (!observation.should_update) return "No BKT observation";
  return observation.outcome === 1 ? "Correct observation" : "Incorrect observation";
}

function ParameterCard({
  symbol,
  title,
  value,
}: {
  symbol: string;
  title: string;
  value?: number | null;
}) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-zinc-50/70 p-3">
      <div className="font-mono text-[10px] font-semibold text-violet-600">{symbol}</div>
      <div className="mt-1 font-mono text-base font-bold text-zinc-950">{number(value)}</div>
      <div className="mt-0.5 text-[10px] leading-4 text-zinc-500">{title}</div>
    </div>
  );
}

function SectionHeading({
  step,
  title,
  detail,
}: {
  step: string;
  title: string;
  detail: string;
}) {
  return (
    <div className="mb-3 flex items-start gap-3">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-zinc-950 text-xs font-bold text-white">
        {step}
      </div>
      <div>
        <div className="text-sm font-semibold text-zinc-950">{title}</div>
        <div className="mt-0.5 text-xs leading-5 text-zinc-500">{detail}</div>
      </div>
    </div>
  );
}

function SignalMeter({
  name,
  value,
  present,
}: {
  name: string;
  value?: number | null;
  present?: boolean | null;
}) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white px-2.5 py-2">
      <div className="flex items-center justify-between gap-2 text-[10px]">
        <span className="font-medium capitalize text-zinc-600">{name}</span>
        <span className="font-mono font-bold text-zinc-900">{number(value)}</span>
      </div>
      <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-zinc-100">
        <div
          className={`h-full rounded-full ${present ? "bg-amber-500" : "bg-zinc-300"}`}
          style={{ width: `${Math.max(0, Math.min(100, (value ?? 0) * 100))}%` }}
        />
      </div>
    </div>
  );
}

function CalculationFlow({
  calculation,
  observation,
}: {
  calculation: BKTCalculationTrace;
  observation: BKTObservationTrace;
}) {
  return (
    <div className="overflow-hidden rounded-xl border border-violet-200 bg-white">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-violet-100 bg-violet-50 px-3 py-2.5">
        <div className="text-xs font-semibold text-violet-900">
          Observation {observation.sequence}: {outcomeLabel(observation)}
        </div>
        <span className="rounded-full bg-white px-2 py-0.5 font-mono text-[10px] font-bold text-violet-700">
          c = {number(observation.update_confidence)}
        </span>
      </div>

      <div className="grid gap-px bg-zinc-100 sm:grid-cols-2">
        <div className="bg-white p-3">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
            A. Bayesian observation
          </div>
          <div className="mt-2 space-y-1.5 font-mono text-[11px] text-zinc-700">
            <div className="flex justify-between gap-3">
              <span>prior mastery</span><strong>{number(calculation.mastery_before)}</strong>
            </div>
            <div className="flex justify-between gap-3">
              <span>P(obs | known)</span><strong>{number(calculation.likelihood_if_known)}</strong>
            </div>
            <div className="flex justify-between gap-3">
              <span>P(obs | not known)</span><strong>{number(calculation.likelihood_if_not_known)}</strong>
            </div>
            <div className="flex justify-between gap-3 border-t border-zinc-100 pt-1.5">
              <span>full posterior</span><strong>{number(calculation.full_observation_posterior)}</strong>
            </div>
          </div>
        </div>

        <div className="bg-white p-3">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
            B. Confidence + learning
          </div>
          <div className="mt-2 space-y-1.5 font-mono text-[11px] text-zinc-700">
            <div className="flex justify-between gap-3">
              <span>weighted posterior</span><strong>{number(calculation.confidence_weighted_posterior)}</strong>
            </div>
            <div className="flex justify-between gap-3">
              <span>effective learn</span><strong>{number(calculation.effective_learn)}</strong>
            </div>
            <div className="flex justify-between gap-3">
              <span>after learning</span><strong>{number(calculation.after_learning)}</strong>
            </div>
            <div className="flex justify-between gap-3 border-t border-zinc-100 pt-1.5">
              <span>after forgetting</span><strong>{number(calculation.after_forgetting)}</strong>
            </div>
          </div>
        </div>
      </div>

      <div className={`flex items-center gap-2 border-t px-3 py-2 text-[10px] font-semibold ${
        calculation.matches_recorded_mastery
          ? "border-emerald-100 bg-emerald-50 text-emerald-700"
          : "border-rose-100 bg-rose-50 text-rose-700"
      }`}>
        {calculation.matches_recorded_mastery ? (
          <CheckCircle2 className="h-3.5 w-3.5" />
        ) : (
          <XCircle className="h-3.5 w-3.5" />
        )}
        Recomputed {number(calculation.after_forgetting)} / recorded {number(calculation.recorded_mastery_after)}
      </div>
    </div>
  );
}

export function BKTUpdateTraceView({ trace }: BKTUpdateTraceViewProps) {
  if (!trace) {
    return (
      <div className="rounded-2xl border border-amber-200 bg-amber-50 p-5 text-sm leading-6 text-amber-800">
        No BKT transaction is checkpointed for this thread yet. Submit a learner
        response, then refresh the trace.
      </div>
    );
  }

  const eventLabel = trace.event_type === "assessment_cycle_bkt_update"
    ? "Formal assessment"
    : "Dialogue turn";
  const applied = trace.observations.filter((item) => item.should_update);
  const p = trace.parameters;
  const masteryDelta = trace.mastery.delta ?? 0;

  return (
    <div className="space-y-5">
      <section className="overflow-hidden rounded-2xl border border-zinc-200 bg-zinc-950 text-white shadow-sm">
        <div className="grid gap-4 p-5 sm:grid-cols-[1fr_auto] sm:items-center">
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-violet-300">
              <Brain className="h-4 w-4" />
              Confidence-weighted BKT
            </div>
            <div className="mt-2 text-xl font-semibold">{trace.target_skill ?? "Target skill"}</div>
            <div className="mt-1 text-xs leading-5 text-zinc-400">
              {eventLabel} · {label(trace.update_timing)} · raw learner content hidden
            </div>
          </div>
          <div className="rounded-xl border border-white/15 bg-white/10 px-4 py-3 text-center">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Mastery</div>
            <div className="mt-1 flex items-center gap-2 font-mono text-sm font-bold">
              {number(trace.mastery.before)}
              <ArrowRight className="h-3.5 w-3.5 text-violet-300" />
              {number(trace.mastery.after)}
            </div>
            <div className={`mt-1 text-[10px] font-semibold ${masteryDelta >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
              Δ {number(trace.mastery.delta)}
            </div>
          </div>
        </div>
        <div className="grid border-t border-white/10 bg-white/[0.04] sm:grid-cols-3">
          <div className="border-b border-white/10 px-4 py-3 sm:border-b-0 sm:border-r">
            <div className="text-[10px] uppercase tracking-wider text-zinc-500">Resolved</div>
            <div className="mt-1 text-xs text-zinc-200">{trace.observations.length} signal(s)</div>
          </div>
          <div className="border-b border-white/10 px-4 py-3 sm:border-b-0 sm:border-r">
            <div className="text-[10px] uppercase tracking-wider text-zinc-500">Applied</div>
            <div className="mt-1 text-xs text-zinc-200">{applied.length} BKT observation(s)</div>
          </div>
          <div className="px-4 py-3">
            <div className="text-[10px] uppercase tracking-wider text-zinc-500">Ledger</div>
            <div className="mt-1 text-xs text-zinc-200">
              {trace.history.observations_before_update ?? "-"} → {trace.history.observations_after_update ?? "-"}
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm">
        <SectionHeading
          step="1"
          title="Load the trained skill model"
          detail="These are the exact per-skill parameters used by the BKT predictor. The effective initial prior is fixed when this learner-skill history begins."
        />
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
          <ParameterCard symbol="P(L₀)" title="population prior" value={p.population_prior} />
          <ParameterCard symbol="P(T)" title="learn transition" value={p.learn} />
          <ParameterCard symbol="P(G)" title="guess" value={p.guess} />
          <ParameterCard symbol="P(S)" title="slip" value={p.slip} />
          <ParameterCard symbol="P(F)" title="forget" value={p.forget} />
        </div>
        <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-violet-200 bg-violet-50 px-3 py-2.5 text-xs">
          <span className="text-violet-800">Effective initial prior</span>
          <span className="font-mono font-bold text-violet-950">
            {number(trace.effective_initial_prior)} · {label(trace.effective_initial_prior_source)}
          </span>
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm">
        <SectionHeading
          step="2"
          title="Resolve evidence into BKT observations"
          detail="The signal resolver decides whether evidence is eligible, assigns a binary outcome, and produces the only confidence consumed by BKT."
        />
        <div className="space-y-3">
          {trace.observations.map((observation) => (
            <div key={observation.event_id ?? observation.sequence} className="overflow-hidden rounded-xl border border-zinc-200">
              <div className={`flex flex-wrap items-center justify-between gap-2 border-b px-3 py-2.5 ${
                observation.should_update
                  ? "border-emerald-100 bg-emerald-50"
                  : "border-zinc-200 bg-zinc-50"
              }`}>
                <div className="flex items-center gap-2 text-xs font-semibold text-zinc-900">
                  {observation.should_update ? (
                    <CheckCircle2 className="h-4 w-4 text-emerald-600" />
                  ) : (
                    <ShieldCheck className="h-4 w-4 text-zinc-500" />
                  )}
                  {outcomeLabel(observation)}
                </div>
                <span className="rounded-full border border-white bg-white px-2 py-0.5 text-[10px] font-semibold text-zinc-600">
                  {label(observation.observation_source)}
                </span>
              </div>

              <div className="grid gap-px bg-zinc-100 sm:grid-cols-2">
                <div className="bg-white p-3 text-[11px]">
                  <div className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-1.5">
                    <span className="text-zinc-500">Primary signal</span>
                    <strong className="text-right text-zinc-900">{label(observation.primary_signal)}</strong>
                    <span className="text-zinc-500">Evidence category</span>
                    <strong className="text-right text-zinc-900">{label(observation.evidence.evidence_category)}</strong>
                    <span className="text-zinc-500">Evidence weight</span>
                    <strong className="font-mono text-zinc-900">{number(observation.evidence_weight)}</strong>
                    <span className="text-zinc-500">Behaviour factor</span>
                    <strong className="font-mono text-zinc-900">{number(observation.behaviour_factor)}</strong>
                    <span className="text-zinc-500">Final update confidence</span>
                    <strong className="font-mono text-violet-700">{number(observation.update_confidence)}</strong>
                  </div>
                  {observation.evidence.reported_evaluator_confidence != null && (
                    <div className="mt-3 flex flex-wrap items-center gap-1.5 rounded-lg bg-zinc-50 px-2.5 py-2 font-mono text-[10px] text-zinc-600">
                      reported {number(observation.evidence.reported_evaluator_confidence)}
                      <ArrowRight className="h-3 w-3" />
                      cap {number(observation.evidence.evaluator_confidence_cap)}
                      <ArrowRight className="h-3 w-3" />
                      applied {number(observation.evidence.applied_evaluator_confidence)}
                    </div>
                  )}
                </div>

                <div className="bg-zinc-50/60 p-3">
                  <div className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-400">Behaviour detectors</div>
                  <div className="grid gap-1.5">
                    <SignalMeter name="reasoning" value={observation.behaviour.reasoning_probability} present={observation.behaviour.reasoning_present} />
                    <SignalMeter name="uncertainty" value={observation.behaviour.uncertainty_probability} present={observation.behaviour.uncertainty_present} />
                    <SignalMeter name="clarification" value={observation.behaviour.clarification_probability} present={observation.behaviour.clarification_present} />
                  </div>
                  <div className="mt-2 text-[10px] text-zinc-500">
                    Contributors: {observation.contributors.length ? observation.contributors.join(", ") : "none"} · repeated misunderstanding: {observation.behaviour.repeated_misunderstanding ? "yes" : "no"}
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between gap-3 border-t border-zinc-100 bg-white px-3 py-2 text-[10px]">
                <span className="truncate font-mono text-zinc-400" title={observation.event_id ?? undefined}>{observation.event_id ?? `observation ${observation.sequence}`}</span>
                <span className="shrink-0 font-semibold text-zinc-600">{label(observation.persistence_status)}</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm">
        <SectionHeading
          step="3"
          title="Apply the confidence-weighted Bayesian update"
          detail="BKT first computes the full observation posterior, blends it with the prior using update confidence, then applies confidence-scaled learning and forgetting."
        />
        {applied.some((item) => item.calculation) ? (
          <div className="space-y-3">
            {applied.map((observation) => observation.calculation && (
              <CalculationFlow
                key={observation.event_id ?? observation.sequence}
                calculation={observation.calculation}
                observation={observation}
              />
            ))}
          </div>
        ) : (
          <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-4 text-sm text-zinc-600">
            No eligible observation was applied, so mastery stayed unchanged and the Bayesian calculation was skipped.
          </div>
        )}
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm">
        <SectionHeading
          step="4"
          title="Persist and audit mastery"
          detail="The resolved-event ledger makes retries idempotent, while independent recomputation checks the stored mastery result."
        />
        <div className="grid gap-2 sm:grid-cols-3">
          <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3">
            <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500"><Database className="h-3.5 w-3.5" /> Observation ledger</div>
            <div className="mt-2 flex items-center gap-2 font-mono text-sm font-bold text-zinc-950">
              {trace.history.observations_before_update ?? "-"}
              <ArrowRight className="h-3.5 w-3.5 text-zinc-400" />
              {trace.history.observations_after_update ?? "-"}
            </div>
            <div className="mt-1 text-[10px] text-zinc-500">{trace.history.observations_applied} applied now</div>
          </div>
          <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3">
            <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500"><Gauge className="h-3.5 w-3.5" /> Stored mastery</div>
            <div className="mt-2 font-mono text-sm font-bold text-zinc-950">{number(trace.persisted_mastery_probability ?? trace.mastery.after)}</div>
            <div className="mt-1 text-[10px] capitalize text-zinc-500">{trace.persisted_mastery_label ?? "unchanged / no label write"}</div>
          </div>
          <div className={`rounded-xl border p-3 ${
            trace.mastery.after_consistent === false
              ? "border-rose-200 bg-rose-50"
              : "border-emerald-200 bg-emerald-50"
          }`}>
            <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500"><ShieldCheck className="h-3.5 w-3.5" /> Audit checks</div>
            <div className="mt-2 text-xs font-bold text-zinc-950">
              {trace.mastery.before_consistent !== false && trace.mastery.after_consistent !== false ? "Consistent" : "Mismatch"}
            </div>
            <div className="mt-1 text-[10px] text-zinc-500">
              history suffix {trace.history.persisted_suffix_matches === false ? "mismatch" : "matches"}
            </div>
          </div>
        </div>
        <div className="mt-3 break-all rounded-lg bg-zinc-50 px-3 py-2 font-mono text-[10px] leading-4 text-zinc-500">
          {trace.schema_version ?? "BKT schema pending"} · {trace.session_id ?? trace.attempt_id ?? "session pending"}
        </div>
      </section>

      <div className="rounded-xl border border-blue-200 bg-blue-50 px-3 py-2.5 text-[10px] leading-4 text-blue-700">
        This tab exposes numerical BKT inputs, decisions, and outputs only. Student text, teacher text, formal answers, expected answers, and evaluator reasoning are intentionally excluded.
      </div>
    </div>
  );
}
