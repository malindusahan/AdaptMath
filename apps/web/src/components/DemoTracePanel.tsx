import { useEffect, useRef, useState } from "react";
import { RefreshCw, X } from "lucide-react";
import { ApiError, getTutorDemoTrace } from "../api/client";
import type { TutorDemoTraceResponse } from "../types/tutor";
import { BKTUpdateTraceView } from "./BKTUpdateTraceView";
import { MoveSelectorTraceView } from "./MoveSelectorTraceView";

interface DemoTracePanelProps {
  open: boolean;
  threadId: string | null;
  onClose: () => void;
}

function compactError(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  if (error instanceof Error) return error.message;
  return "Unable to load demo diagnostics.";
}

function score(value?: number | null): string {
  return typeof value === "number" ? value.toFixed(4) : "Pending";
}

function displayLabel(value?: string | null): string {
  if (!value) return "Pending";
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function StatePill({
  label,
  tone = "neutral",
}: {
  label: string;
  tone?: "good" | "info" | "warn" | "neutral";
}) {
  const classes =
    tone === "good"
      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
      : tone === "info"
        ? "border-blue-200 bg-blue-50 text-blue-700"
        : tone === "warn"
          ? "border-amber-200 bg-amber-50 text-amber-700"
          : "border-zinc-200 bg-zinc-50 text-zinc-600";

  return (
    <span
      className={`inline-flex rounded-full border px-2 py-0.5 text-[11px] font-semibold ${classes}`}
    >
      {label}
    </span>
  );
}

function StageCard({
  index,
  title,
  status,
  result,
  detail,
}: {
  index: number;
  title: string;
  status: string;
  result: string;
  detail: string;
}) {
  const tone =
    status === "Complete" || status === "Used" || status === "Ready"
      ? "good"
      : status === "Skipped"
        ? "neutral"
        : "info";

  return (
    <div className="rounded-xl border border-zinc-200 bg-white p-3 shadow-sm">
      <div className="flex items-start gap-3">
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-zinc-900 text-xs font-bold text-white">
          {index}
        </div>
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <div className="text-sm font-semibold text-zinc-900">{title}</div>
            <StatePill label={status} tone={tone} />
          </div>
          <div className="mt-1.5 text-sm font-semibold text-zinc-800">
            {result}
          </div>
          <div className="mt-1 text-xs leading-5 text-zinc-500">{detail}</div>
        </div>
      </div>
    </div>
  );
}

export function DemoTracePanel({
  open,
  threadId,
  onClose,
}: DemoTracePanelProps) {
  const [trace, setTrace] = useState<TutorDemoTraceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<
    "workflow" | "move-selector" | "bkt"
  >("workflow");
  const traceRequestSequence = useRef(0);

  async function refreshTrace(clearExisting = false) {
    if (!threadId) return;
    const requestSequence = ++traceRequestSequence.current;
    if (clearExisting) setTrace(null);
    setLoading(true);
    setError(null);
    try {
      const nextTrace = await getTutorDemoTrace(threadId);
      if (requestSequence === traceRequestSequence.current) {
        setTrace(nextTrace);
      }
    } catch (caught) {
      if (requestSequence === traceRequestSequence.current) {
        setError(compactError(caught));
      }
    } finally {
      if (requestSequence === traceRequestSequence.current) {
        setLoading(false);
      }
    }
  }

  useEffect(() => {
    if (!open || !threadId) {
      traceRequestSequence.current += 1;
      setLoading(false);
      if (!threadId) setTrace(null);
      return;
    }
    void refreshTrace(true);
  }, [open, threadId]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[80]">
      <button
        type="button"
        className="absolute inset-0 bg-zinc-950/20 backdrop-blur-[1px]"
        aria-label="Close demo trace"
        onClick={onClose}
      />

      <aside
        className="absolute inset-y-0 right-0 flex w-full max-w-2xl flex-col border-l border-zinc-200 bg-zinc-50 shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby="demo-trace-title"
      >
        <header className="flex shrink-0 items-center justify-between border-b border-zinc-200 bg-white px-5 py-4">
          <div>
            <div id="demo-trace-title" className="text-base font-semibold text-zinc-950">
              AdaptMath Demo Trace
            </div>
            <div className="mt-0.5 text-xs text-zinc-500">
              Live lesson cycle, move selector, and BKT updates
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => void refreshTrace()}
              disabled={!threadId || loading}
              className="inline-flex items-center gap-1.5 rounded-lg border border-zinc-200 bg-white px-3 py-2 text-xs font-semibold text-zinc-700 hover:bg-zinc-50 disabled:opacity-40"
            >
              <RefreshCw
                className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`}
              />
              Refresh
            </button>
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-100"
              aria-label="Close demo trace"
            >
              <X className="h-5 w-5" />
            </button>
          </div>
        </header>

        <div className="flex shrink-0 border-b border-zinc-200 bg-white px-5">
          <button
            type="button"
            onClick={() => setActiveTab("workflow")}
            className={`border-b-2 px-3 py-3 text-xs font-semibold transition-colors ${
              activeTab === "workflow"
                ? "border-zinc-950 text-zinc-950"
                : "border-transparent text-zinc-500 hover:text-zinc-800"
            }`}
          >
            Lesson cycle
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("move-selector")}
            className={`border-b-2 px-3 py-3 text-xs font-semibold transition-colors ${
              activeTab === "move-selector"
                ? "border-blue-600 text-blue-700"
                : "border-transparent text-zinc-500 hover:text-zinc-800"
            }`}
          >
            Move selector
          </button>
          <button
            type="button"
            onClick={() => setActiveTab("bkt")}
            className={`border-b-2 px-3 py-3 text-xs font-semibold transition-colors ${
              activeTab === "bkt"
                ? "border-violet-600 text-violet-700"
                : "border-transparent text-zinc-500 hover:text-zinc-800"
            }`}
          >
            BKT updates
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-5">
          {!threadId && (
            <div className="rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
              Start an adaptive maths lesson first. The trace reads the real
              persisted Tutor thread.
            </div>
          )}

          {error && (
            <div
              className="mb-4 rounded-xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700"
              role="alert"
            >
              {error}
            </div>
          )}

          {loading && !trace && (
            <div className="rounded-xl border border-zinc-200 bg-white p-5 text-sm text-zinc-500">
              Reading the persisted lesson state...
            </div>
          )}

          {trace && activeTab === "workflow" && (
            <div className="space-y-4">
              <section className="rounded-2xl bg-zinc-950 p-5 text-white shadow-sm">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="text-xs font-semibold uppercase tracking-[0.16em] text-zinc-400">
                      Current lesson cycle
                    </div>
                    <div className="mt-1 text-lg font-semibold">
                      CHECK &rarr; DECIDE &rarr; TEACH &rarr; CHECK &rarr; REPAIR
                    </div>
                  </div>
                  <StatePill label={displayLabel(trace.phase)} tone="info" />
                </div>

              </section>

              <section>
                <div className="mb-2 text-sm font-semibold text-zinc-950">
                  What happened in this cycle
                </div>
                <div className="grid gap-2">
                  <StageCard
                    index={1}
                    title="Check problem"
                    status="Complete"
                    result={`Complexity ${score(trace.complexity_score)}`}
                    detail={`Target skill: ${displayLabel(trace.target_skill)}`}
                  />
                  <StageCard
                    index={2}
                    title="Decide teaching path"
                    status="Complete"
                    result={displayLabel(trace.route)}
                    detail={
                      trace.planner_used
                        ? "Router included the Planner before tutoring."
                        : "Router sent the lesson directly to the Tutor."
                    }
                  />
                  <StageCard
                    index={3}
                    title="Teach"
                    status="Complete"
                    result={`${displayLabel(
                      trace.selected_arm ?? trace.pedagogical_move,
                    )} - Tutor turn ${trace.turn_count}`}
                    detail={
                      trace.move_overridden
                        ? "The recorded Tutor move differs from the selector handoff."
                        : "The Tutor executed the move-selector decision unchanged."
                    }
                  />
                  <StageCard
                    index={4}
                    title="Check understanding"
                    status={trace.evaluation_available ? "Complete" : "Current"}
                    result={displayLabel(trace.teaching_status)}
                    detail={
                      trace.evaluation_available
                        ? `${trace.correct_answer_count} correct - ${trace.wrong_answer_count} incorrect`
                        : trace.assessment_question_count > 0
                          ? `Assessment progress: ${trace.assessment_question_count}/3 questions`
                          : "The latest student response determines the next step."
                    }
                  />
                  <StageCard
                    index={5}
                    title="Repair or continue"
                    status={trace.needs_reteaching ? "Current" : "Ready"}
                    result={
                      trace.needs_reteaching
                        ? `Targeted reteaching - round ${Math.max(1, trace.reteach_round)}`
                        : trace.next_nodes.length
                          ? displayLabel(trace.next_nodes[0])
                          : "Lesson complete"
                    }
                    detail={
                      trace.needs_reteaching
                        ? "The next cycle targets the detected learning gap."
                        : trace.teaching_status === "continue_teaching"
                          ? "Waiting for the next student response."
                          : "Continue to the next recorded lesson action."
                    }
                  />
                </div>
              </section>

              <section className="rounded-xl border border-blue-200 bg-blue-50 p-3 text-xs leading-5 text-blue-800">
                Complexity and routing are separate. The complexity model only
                outputs a continuous score. This tab shows the Router's selected
                result; move-selection and BKT internals stay in their dedicated
                tabs.
              </section>
            </div>
          )}

          {trace && activeTab === "move-selector" && (
            <MoveSelectorTraceView trace={trace.move_selector} />
          )}

          {trace && activeTab === "bkt" && (
            <BKTUpdateTraceView trace={trace.bkt_update} />
          )}
        </div>
      </aside>
    </div>
  );
}
