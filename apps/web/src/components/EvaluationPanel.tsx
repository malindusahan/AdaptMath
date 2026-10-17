import { CheckCircle2, RotateCcw, TriangleAlert } from "lucide-react";
import type { EvaluationResult } from "../types/tutor";

interface EvaluationPanelProps {
  evaluation: EvaluationResult;
}

export function EvaluationPanel({ evaluation }: EvaluationPanelProps) {
  return (
    <section className="rounded-3xl border border-slate-200 bg-white p-5 shadow-sm sm:p-6">
      <div className="flex items-start gap-3">
        <div
          className={`rounded-xl p-2 ${
            evaluation.needs_reteaching
              ? "bg-amber-100 text-amber-700"
              : "bg-emerald-100 text-emerald-700"
          }`}
        >
          {evaluation.needs_reteaching ? (
            <RotateCcw className="h-5 w-5" aria-hidden="true" />
          ) : (
            <CheckCircle2 className="h-5 w-5" aria-hidden="true" />
          )}
        </div>

        <div className="min-w-0 flex-1">
          <h2 className="font-bold text-slate-950">
            {evaluation.needs_reteaching
              ? "AdaptMath adjusted the lesson"
              : "Understanding confirmed"}
          </h2>
          {evaluation.overall_feedback && (
            <p className="mt-1 text-sm leading-6 text-slate-600">
              {evaluation.overall_feedback}
            </p>
          )}
        </div>
      </div>

      {evaluation.identified_errors.length > 0 && (
        <div className="mt-5 rounded-2xl bg-amber-50 p-4">
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-amber-900">
            <TriangleAlert className="h-4 w-4" aria-hidden="true" />
            What to work on
          </div>
          <ul className="grid gap-2 text-sm leading-6 text-amber-950/80">
            {evaluation.identified_errors.map((error) => (
              <li key={error} className="flex gap-2">
                <span aria-hidden="true">•</span>
                <span>{error}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
