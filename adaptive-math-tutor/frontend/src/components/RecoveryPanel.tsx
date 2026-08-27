import { RefreshCw, ShieldAlert } from "lucide-react";

interface RecoveryPanelProps {
  disabled?: boolean;
  onResume: () => void;
}

export function RecoveryPanel({ disabled = false, onResume }: RecoveryPanelProps) {
  return (
    <section className="rounded-3xl border border-amber-200 bg-amber-50 p-5 sm:p-6">
      <div className="flex items-start gap-3">
        <div className="rounded-xl bg-amber-100 p-2 text-amber-700">
          <ShieldAlert className="h-5 w-5" aria-hidden="true" />
        </div>
        <div className="flex-1">
          <h2 className="font-bold text-amber-950">Your lesson is safe</h2>
          <p className="mt-1 text-sm leading-6 text-amber-900/80">
            The tutoring workflow paused before completing a backend step. Your progress was saved and can be resumed from the last durable checkpoint.
          </p>
          <button
            type="button"
            onClick={onResume}
            disabled={disabled}
            className="mt-4 inline-flex items-center gap-2 rounded-xl bg-amber-950 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-amber-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${disabled ? "animate-spin" : ""}`} aria-hidden="true" />
            Resume lesson
          </button>
        </div>
      </div>
    </section>
  );
}
