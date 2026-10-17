import { useState, type FormEvent } from "react";
import { ArrowUp, Sparkles } from "lucide-react";

interface QuestionComposerProps {
  disabled?: boolean;
  onSubmit: (question: string) => void;
}

export function QuestionComposer({ disabled = false, onSubmit }: QuestionComposerProps) {
  const [question, setQuestion] = useState("");

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const cleaned = question.trim();
    if (!cleaned || disabled) return;
    onSubmit(cleaned);
  }

  return (
    <form onSubmit={submit} className="rounded-3xl border border-slate-200 bg-white p-3 shadow-lg shadow-slate-200/40">
      <label className="sr-only" htmlFor="math-question">
        Ask a mathematics question
      </label>
      <textarea
        id="math-question"
        value={question}
        onChange={(event) => setQuestion(event.target.value)}
        disabled={disabled}
        rows={4}
        maxLength={4000}
        placeholder="Ask a maths question… e.g. Solve 3x + 5 = 20"
        className="min-h-28 w-full resize-none rounded-2xl border-0 bg-slate-50 px-4 py-4 text-base leading-7 text-slate-900 outline-none placeholder:text-slate-400 focus:bg-white focus:ring-2 focus:ring-indigo-200 disabled:cursor-not-allowed disabled:opacity-60"
      />

      <div className="mt-3 flex items-center justify-between gap-3 px-1 pb-1">
        <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
          <Sparkles className="h-3.5 w-3.5 text-indigo-500" aria-hidden="true" />
          AdaptMath decides the support path behind the scenes.
        </div>
        <button
          type="submit"
          disabled={disabled || question.trim().length === 0}
          className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-slate-950 text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-slate-300"
          aria-label="Send question"
        >
          <ArrowUp className="h-4 w-4" aria-hidden="true" />
        </button>
      </div>
    </form>
  );
}
