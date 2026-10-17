import { useEffect, useRef } from "react";
import { AlertTriangle, LogOut, Trash2, X } from "lucide-react";

export interface ConfirmationOptions {
  title: string;
  message: string;
  confirmLabel: string;
  cancelLabel?: string;
  tone?: "warning" | "danger";
  icon?: "warning" | "delete" | "logout";
}

interface ConfirmationDialogProps {
  confirmation: ConfirmationOptions | null;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmationDialog({
  confirmation,
  onConfirm,
  onCancel,
}: ConfirmationDialogProps) {
  const cancelButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!confirmation) return;

    cancelButtonRef.current?.focus();
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onCancel();
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [confirmation, onCancel]);

  if (!confirmation) return null;

  const destructive = confirmation.tone === "danger";
  const Icon =
    confirmation.icon === "delete"
      ? Trash2
      : confirmation.icon === "logout"
        ? LogOut
        : AlertTriangle;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center px-4 py-8">
      <button
        type="button"
        className="confirmation-backdrop absolute inset-0 cursor-default bg-slate-950/55 backdrop-blur-sm"
        aria-label="Close confirmation"
        onClick={onCancel}
      />

      <section
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="confirmation-title"
        aria-describedby="confirmation-message"
        className="confirmation-card relative w-full max-w-sm overflow-hidden rounded-2xl border border-white/80 bg-white p-5 shadow-2xl shadow-slate-950/25 dark:border-slate-700 dark:bg-slate-900"
      >
        <button
          type="button"
          onClick={onCancel}
          className="absolute right-3 top-3 rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-700 dark:hover:bg-slate-800 dark:hover:text-slate-200"
          aria-label="Close confirmation"
        >
          <X className="h-4 w-4" />
        </button>

        <div
          className={`flex h-10 w-10 items-center justify-center rounded-xl ${
            destructive
              ? "bg-rose-50 text-rose-600 dark:bg-rose-500/10 dark:text-rose-300"
              : "bg-amber-50 text-amber-600 dark:bg-amber-500/10 dark:text-amber-300"
          }`}
        >
          <Icon className="h-4.5 w-4.5" aria-hidden="true" />
        </div>

        <h2
          id="confirmation-title"
          className="mt-4 pr-8 text-lg font-extrabold tracking-tight text-slate-950 dark:text-white"
        >
          {confirmation.title}
        </h2>
        <p
          id="confirmation-message"
          className="mt-1.5 text-sm leading-5.5 text-slate-600 dark:text-slate-300"
        >
          {confirmation.message}
        </p>

        <div className="mt-5 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
          <button
            ref={cancelButtonRef}
            type="button"
            onClick={onCancel}
            className="rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-bold text-slate-700 transition hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700"
          >
            {confirmation.cancelLabel ?? "Keep working"}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className={`rounded-xl px-3.5 py-2 text-xs font-bold text-white shadow-sm transition hover:-translate-y-0.5 ${
              destructive
                ? "bg-rose-600 hover:bg-rose-500"
                : "bg-indigo-600 hover:bg-indigo-500"
            }`}
          >
            {confirmation.confirmLabel}
          </button>
        </div>
      </section>
    </div>
  );
}
