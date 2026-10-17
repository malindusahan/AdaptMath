import { useEffect, useState, type FormEvent } from "react";
import { ClipboardCheck, Send } from "lucide-react";
import type { AssessmentQuestion, TutorAnswerPayload } from "../types/tutor";

interface AssessmentPanelProps {
  questions: AssessmentQuestion[];
  message?: string | null;
  disabled?: boolean;
  onSubmit: (payload: TutorAnswerPayload) => void;
}

export function AssessmentPanel({
  questions,
  message,
  disabled = false,
  onSubmit,
}: AssessmentPanelProps) {
  const [answers, setAnswers] = useState<Record<string, string>>({});

  useEffect(() => {
    setAnswers({});
  }, [questions]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();

    const payload = questions.map((question) => ({
      question_id: question.question_id,
      answer: (answers[question.question_id] ?? "").trim(),
    }));

    if (payload.some((answer) => !answer.answer)) return;

    onSubmit({ answers: payload });
  }

  const complete =
    questions.length > 0 &&
    questions.every((question) => (answers[question.question_id] ?? "").trim().length > 0);

  return (
    <form
      onSubmit={submit}
      className="rounded-3xl border border-indigo-100 bg-indigo-50/70 p-5 shadow-sm sm:p-6"
    >
      <div className="mb-5 flex items-start gap-3">
        <div className="rounded-xl bg-indigo-600 p-2 text-white">
          <ClipboardCheck className="h-5 w-5" aria-hidden="true" />
        </div>
        <div>
          <h2 className="font-bold text-slate-950">Quick understanding check</h2>
          <p className="mt-1 text-sm leading-6 text-slate-600">
            {message || "Answer all three questions so AdaptMath can decide what support you need next."}
          </p>
        </div>
      </div>

      <div className="grid gap-5">
        {questions.map((question, index) => (
          <label key={question.question_id} className="grid gap-2">
            <span className="text-sm font-semibold text-slate-800">
              {index + 1}. {question.question}
            </span>
            <textarea
              value={answers[question.question_id] ?? ""}
              onChange={(event) =>
                setAnswers((current) => ({
                  ...current,
                  [question.question_id]: event.target.value,
                }))
              }
              disabled={disabled}
              required
              rows={3}
              maxLength={2000}
              className="min-h-24 resize-y rounded-2xl border border-indigo-100 bg-white px-4 py-3 text-sm leading-6 text-slate-900 outline-none transition focus:border-indigo-300 focus:ring-4 focus:ring-indigo-100 disabled:cursor-not-allowed disabled:opacity-60"
              placeholder="Write your answer…"
            />
          </label>
        ))}
      </div>

      <button
        type="submit"
        disabled={disabled || !complete}
        className="primary-button mt-5"
      >
        Submit answers
        <Send className="h-4 w-4" aria-hidden="true" />
      </button>
    </form>
  );
}
