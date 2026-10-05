import {
  ArrowRight,
  BrainCircuit,
  CheckCircle2,
  Clock3,
  Database,
} from "lucide-react";
import type {
  MoveSelectorContextFeature,
  MoveSelectorTrace,
} from "../types/tutor";

interface MoveSelectorTraceViewProps {
  trace: MoveSelectorTrace | null | undefined;
}

const MOVE_ORDER = ["generic", "probing", "focus", "telling"];

const BLOCK_META = {
  S: {
    title: "S - Selector signal",
    detail: "Four probabilities from frozen MD7.",
    classes: "border-blue-200 bg-blue-50/60 text-blue-800",
  },
  K: {
    title: "K - Knowledge state",
    detail: "BKT mastery and the previous mastery change.",
    classes: "border-violet-200 bg-violet-50/60 text-violet-800",
  },
  L: {
    title: "L - Learner signals",
    detail: "Signals observed before this action, never from the future turn.",
    classes: "border-amber-200 bg-amber-50/60 text-amber-800",
  },
  H: {
    title: "H - Move history",
    detail: "Logged diagnostic context; not active in the current 11D policy.",
    classes: "border-zinc-200 bg-zinc-50 text-zinc-700",
  },
  Q: {
    title: "Q - Response quality",
    detail: "Logged diagnostic context; not active in the current 11D policy.",
    classes: "border-zinc-200 bg-zinc-50 text-zinc-700",
  },
  unknown: {
    title: "Other",
    detail: "Additional selector context.",
    classes: "border-zinc-200 bg-zinc-50 text-zinc-700",
  },
} as const;

function number(value: number | null | undefined, digits = 4): string {
  return typeof value === "number" ? value.toFixed(digits) : "Pending";
}

function moveLabel(move: string): string {
  return move.charAt(0).toUpperCase() + move.slice(1);
}

function FeatureRow({ feature }: { feature: MoveSelectorContextFeature }) {
  const indicatorState = feature.missing_indicator
    ? feature.value >= 0.5
      ? "missing"
      : "available"
    : null;

  return (
    <div className="grid grid-cols-[2rem_minmax(0,1fr)_5.25rem] items-center gap-3 border-t border-zinc-100 px-3 py-2.5 first:border-t-0">
      <div className="font-mono text-[11px] text-zinc-400">
        {String(feature.index).padStart(2, "0")}
      </div>
      <div className="min-w-0">
        <div className="break-words font-mono text-xs font-semibold text-zinc-800">
          {feature.name}
        </div>
        <div className="mt-0.5 text-[11px] leading-4 text-zinc-500">
          {feature.description}
        </div>
      </div>
      <div className="text-right">
        <div className="font-mono text-xs font-bold text-zinc-950">
          {number(feature.value)}
        </div>
        {indicatorState && (
          <div
            className={`mt-0.5 text-[10px] font-semibold ${
              indicatorState === "missing" ? "text-amber-700" : "text-emerald-700"
            }`}
          >
            {indicatorState}
          </div>
        )}
      </div>
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

export function MoveSelectorTraceView({ trace }: MoveSelectorTraceViewProps) {
  if (!trace) {
    return (
      <div className="rounded-2xl border border-amber-200 bg-amber-50 p-5 text-sm leading-6 text-amber-800">
        No direct Turn-LinTS decision is stored for this thread yet. Start an
        adaptive Tutor turn, then refresh the trace.
      </div>
    );
  }

  const grouped = trace.context_features.reduce<Record<string, MoveSelectorContextFeature[]>>(
    (result, feature) => {
      (result[feature.block] ??= []).push(feature);
      return result;
    },
    {},
  );
  const rankedArms = [...trace.available_arms].sort(
    (left, right) =>
      (trace.policy_scores[right] ?? trace.sampled_scores[right] ?? -Infinity) -
      (trace.policy_scores[left] ?? trace.sampled_scores[left] ?? -Infinity),
  );
  const reward = trace.latest_reward_update;
  const rewardMatchesCurrent =
    Boolean(reward?.action_event_id) && reward?.action_event_id === trace.action_event_id;

  return (
    <div className="space-y-5">
      <section className="overflow-hidden rounded-2xl border border-zinc-200 bg-zinc-950 text-white shadow-sm">
        <div className="grid gap-4 p-5 sm:grid-cols-[1fr_auto] sm:items-center">
          <div>
            <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-blue-300">
              <BrainCircuit className="h-4 w-4" />
              Direct-arm Turn-LinTS
            </div>
            <div className="mt-2 text-xl font-semibold">
              {trace.context_dimension}D context to one teaching move
            </div>
            <div className="mt-1 max-w-xl text-xs leading-5 text-zinc-400">
              The decision uses only information available before the Tutor
              response. The chosen arm is sent to the Tutor unchanged.
            </div>
          </div>
          <div className="rounded-xl border border-white/15 bg-white/10 px-4 py-3 text-center">
            <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-400">
              Final move
            </div>
            <div className="mt-1 text-lg font-bold text-white">
              {trace.selected_move ? moveLabel(trace.selected_move) : "Pending"}
            </div>
          </div>
        </div>
        <div className="grid border-t border-white/10 bg-white/[0.04] sm:grid-cols-3">
          <div className="border-b border-white/10 px-4 py-3 sm:border-b-0 sm:border-r">
            <div className="text-[10px] uppercase tracking-wider text-zinc-500">Mode</div>
            <div className="mt-1 font-mono text-xs text-zinc-200">{trace.policy_mode ?? "Pending"}</div>
          </div>
          <div className="border-b border-white/10 px-4 py-3 sm:border-b-0 sm:border-r">
            <div className="text-[10px] uppercase tracking-wider text-zinc-500">Context</div>
            <div className="mt-1 font-mono text-xs text-zinc-200">
              {trace.enabled_blocks.join("+") || "Pending"} / {trace.context_dimension}D
            </div>
          </div>
          <div className="px-4 py-3">
            <div className="text-[10px] uppercase tracking-wider text-zinc-500">Action</div>
            <div className="mt-1 truncate font-mono text-xs text-zinc-200" title={trace.action_event_id ?? undefined}>
              turn {trace.action_turn_index ?? "-"}
            </div>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm">
        <SectionHeading
          step="1"
          title="Frozen MD7 signal"
          detail="MD7 supplies a probability vector. Its argmax is an input, not the final policy decision."
        />
        <div className="grid gap-2 sm:grid-cols-2">
          {MOVE_ORDER.map((move) => {
            const probability = trace.md7_probabilities[move];
            const isBase = move === trace.base_model_move;
            return (
              <div
                key={move}
                className={`rounded-xl border p-3 ${
                  isBase ? "border-blue-300 bg-blue-50" : "border-zinc-200 bg-zinc-50/60"
                }`}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="text-xs font-semibold text-zinc-800">{moveLabel(move)}</div>
                  <div className="font-mono text-xs font-bold text-zinc-950">{number(probability)}</div>
                </div>
                <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-zinc-200">
                  <div
                    className={`h-full rounded-full ${isBase ? "bg-blue-600" : "bg-zinc-500"}`}
                    style={{ width: `${Math.max(0, Math.min(100, (probability ?? 0) * 100))}%` }}
                  />
                </div>
                <div className="mt-1.5 text-[10px] font-medium text-zinc-500">
                  {isBase ? "MD7 argmax" : "Candidate signal"}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm">
        <SectionHeading
          step="2"
          title={`Build the ${trace.context_dimension}D policy context`}
          detail="Features are ordered exactly as they enter Turn-LinTS. Missing values use explicit indicator features."
        />
        <div className="space-y-3">
          {trace.enabled_blocks.map((block) => {
            const meta = BLOCK_META[block as keyof typeof BLOCK_META] ?? BLOCK_META.unknown;
            return (
              <div key={block} className="overflow-hidden rounded-xl border border-zinc-200">
                <div className={`border-b px-3 py-2.5 ${meta.classes}`}>
                  <div className="text-xs font-bold">{meta.title}</div>
                  <div className="mt-0.5 text-[11px] opacity-80">{meta.detail}</div>
                </div>
                <div className="bg-white">
                  {(grouped[block] ?? []).map((feature) => (
                    <FeatureRow key={feature.index} feature={feature} />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
        <div className="mt-3 break-all rounded-lg bg-zinc-50 px-3 py-2 font-mono text-[10px] leading-4 text-zinc-500">
          schema: {trace.context_schema_id ?? "Pending"}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm">
        <SectionHeading
          step="3"
          title="Sample and select a direct arm"
          detail="Turn-LinTS samples an expected reward for every available move and selects the highest policy score."
        />
        <div className="overflow-hidden rounded-xl border border-zinc-200">
          <div className="grid grid-cols-[1fr_7rem_7rem] bg-zinc-50 px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-zinc-500">
            <div>Arm</div>
            <div className="text-right">TS sample</div>
            <div className="text-right">Policy score</div>
          </div>
          {rankedArms.map((arm) => {
            const selected = arm === trace.selected_move;
            return (
              <div
                key={arm}
                className={`grid grid-cols-[1fr_7rem_7rem] items-center border-t px-3 py-3 text-xs ${
                  selected ? "border-emerald-200 bg-emerald-50" : "border-zinc-100"
                }`}
              >
                <div className="flex items-center gap-2 font-semibold text-zinc-900">
                  {selected && <CheckCircle2 className="h-4 w-4 text-emerald-600" />}
                  {moveLabel(arm)}
                  {selected && (
                    <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[10px] text-emerald-700">
                      selected
                    </span>
                  )}
                </div>
                <div className="text-right font-mono text-zinc-700">
                  {number(trace.sampled_scores[arm])}
                </div>
                <div className="text-right font-mono font-bold text-zinc-950">
                  {number(trace.policy_scores[arm] ?? trace.sampled_scores[arm])}
                </div>
              </div>
            );
          })}
        </div>

        <div className="mt-3 flex flex-wrap items-center justify-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-sm">
          <span className="text-zinc-600">Turn-LinTS</span>
          <strong className="text-zinc-950">{trace.selected_move ?? "Pending"}</strong>
          <ArrowRight className="h-4 w-4 text-emerald-600" />
          <span className="text-zinc-600">Tutor receives</span>
          <strong className="text-zinc-950">{trace.tutor_move ?? "Pending"}</strong>
          <span
            className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
              trace.post_selection_override
                ? "bg-rose-100 text-rose-700"
                : "bg-emerald-100 text-emerald-700"
            }`}
          >
            {trace.post_selection_override ? "contract failed" : "unchanged"}
          </span>
        </div>
        <div className="mt-2 text-center text-[10px] text-zinc-500">
          Decision source: {trace.decision_source ?? "Pending"} / {trace.behavior_policy ?? "Pending"}
        </div>
      </section>

      <section className="rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm">
        <SectionHeading
          step="4"
          title="Resolve reward and update the posterior"
          detail="The latest resolved student response supplies mastery change and credit for the arm that was actually taught."
        />
        {!reward ? (
          <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
            <Clock3 className="mt-0.5 h-4 w-4 shrink-0" />
            <div>
              <div className="font-semibold">Reward pending</div>
              <div className="mt-1 text-xs leading-5">
                This action needs the learner's next response and a resolved BKT observation.
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="flex items-start gap-3 rounded-xl border border-blue-200 bg-blue-50 p-3 text-xs text-blue-800">
              {rewardMatchesCurrent ? (
                <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" />
              ) : (
                <Clock3 className="mt-0.5 h-4 w-4 shrink-0" />
              )}
              <div>
                <div className="font-semibold">
                  {rewardMatchesCurrent
                    ? "Reward belongs to the current displayed action"
                    : "Latest resolved reward belongs to the previous action"}
                </div>
                <div className="mt-0.5 break-all font-mono text-[10px] opacity-80">
                  {reward.action_event_id}
                </div>
              </div>
            </div>

            <div className="grid gap-2 sm:grid-cols-3">
              <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Mastery</div>
                <div className="mt-2 flex items-center gap-2 font-mono text-sm font-bold text-zinc-950">
                  {number(reward.mastery_before)} <ArrowRight className="h-3.5 w-3.5 text-zinc-400" /> {number(reward.mastery_after)}
                </div>
                <div className="mt-1 text-[10px] text-zinc-500">raw delta {number(reward.raw_delta)}</div>
              </div>
              <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Configured reward</div>
                <div className="mt-2 font-mono text-sm font-bold text-zinc-950">{number(reward.reward_value)}</div>
                <div className="mt-1 text-[10px] text-zinc-500">{reward.reward_mode ?? trace.reward_mode ?? "Pending"}</div>
              </div>
              <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Credit</div>
                <div className="mt-2 text-sm font-bold text-zinc-950">{reward.attributed_move ?? "Pending"}</div>
                <div className="mt-1 text-[10px] text-zinc-500">weight {number(reward.posterior_update_weight, 2)}</div>
              </div>
            </div>

            <div className={`flex items-center gap-3 rounded-xl border p-3 ${
              reward.posterior_updated
                ? "border-emerald-200 bg-emerald-50"
                : "border-zinc-200 bg-zinc-50"
            }`}>
              <Database className={`h-5 w-5 ${reward.posterior_updated ? "text-emerald-600" : "text-zinc-400"}`} />
              <div className="min-w-0 flex-1">
                <div className="text-xs font-semibold text-zinc-900">
                  {reward.posterior_updated ? "Posterior updated" : "No posterior update"}
                </div>
                <div className="mt-0.5 text-[10px] text-zinc-500">
                  {reward.resolution_status ?? "Pending resolution"}
                </div>
              </div>
              <div className="font-mono text-xs font-bold text-zinc-900">
                {reward.posterior_total_updates_before ?? "-"}
                <ArrowRight className="mx-1 inline h-3 w-3 text-zinc-400" />
                {reward.posterior_total_updates_after ?? "-"}
              </div>
            </div>
          </div>
        )}
      </section>

      {Object.keys(trace.response_quality_scores).length > 0 && (
        <section className="rounded-xl border border-zinc-200 bg-zinc-100/70 p-3">
          <div className="text-xs font-semibold text-zinc-800">Tutor response quality</div>
          <div className="mt-2 flex flex-wrap gap-2">
            {Object.entries(trace.response_quality_scores).map(([name, value]) => (
              <span key={name} className="rounded-lg border border-zinc-200 bg-white px-2.5 py-1.5 text-[10px] text-zinc-600">
                {name.replaceAll("_", " ")} <strong className="ml-1 font-mono text-zinc-900">{number(value)}</strong>
              </span>
            ))}
          </div>
          <div className="mt-2 text-[10px] leading-4 text-zinc-500">
            Logged after generation for evaluation. These Q signals are not part of the active 11D S+K+L decision context.
          </div>
        </section>
      )}
    </div>
  );
}
