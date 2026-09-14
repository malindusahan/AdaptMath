import { BrainCircuit, LoaderCircle } from "lucide-react";

interface LoadingCardProps {
  label?: string;
}

export function LoadingCard({ label = "AdaptMath is preparing your lesson…" }: LoadingCardProps) {
  return (
    <div className="rounded-3xl border border-indigo-100 bg-white p-6 shadow-sm">
      <div className="flex items-center gap-4">
        <div className="relative flex h-11 w-11 items-center justify-center rounded-2xl bg-indigo-50 text-indigo-600">
          <BrainCircuit className="h-5 w-5" aria-hidden="true" />
          <LoaderCircle className="absolute -right-1 -top-1 h-4 w-4 animate-spin text-indigo-500" aria-hidden="true" />
        </div>
        <div>
          <p className="font-semibold text-slate-900">Working on your question</p>
          <p className="mt-1 text-sm text-slate-500">{label}</p>
        </div>
      </div>
    </div>
  );
}
