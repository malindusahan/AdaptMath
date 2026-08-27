import { CheckCircle2, LoaderCircle, TriangleAlert } from "lucide-react";

interface StatusPillProps {
  state: "ready" | "checking" | "offline";
}

export function StatusPill({ state }: StatusPillProps) {
  const config = {
    ready: {
      label: "Tutor ready",
      icon: CheckCircle2,
      className: "bg-emerald-50 text-emerald-700 ring-emerald-200",
    },
    checking: {
      label: "Checking tutor",
      icon: LoaderCircle,
      className: "bg-amber-50 text-amber-700 ring-amber-200",
    },
    offline: {
      label: "Tutor unavailable",
      icon: TriangleAlert,
      className: "bg-rose-50 text-rose-700 ring-rose-200",
    },
  } as const;

  const item = config[state];
  const Icon = item.icon;

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-semibold ring-1 ${item.className}`}
    >
      <Icon
        className={`h-3.5 w-3.5 ${state === "checking" ? "animate-spin" : ""}`}
        aria-hidden="true"
      />
      {item.label}
    </span>
  );
}
