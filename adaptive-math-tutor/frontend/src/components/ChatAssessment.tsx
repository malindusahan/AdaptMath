import { useState, type FormEvent } from "react";
import { ArrowUp, CheckCircle2 } from "lucide-react";
import type { ChatMessage } from "../types/chat";
import type { TutorAnswerPayload } from "../types/tutor";

interface ChatAssessmentProps {
  message: Extract<ChatMessage, { type: "assessment" }>;
  disabled?: boolean;
  onSubmit: (payload: TutorAnswerPayload) => void;
}

export function ChatAssessment({ message, disabled = false, onSubmit }: ChatAssessmentProps) {
  const [answers, setAnswers] = useState<Record<string, string>>(() =>
    Object.fromEntries((message.answers ?? []).map((answer) => [answer.question_id, answer.answer])),
  );

  const complete = message.questions.every(
    (question) => (answers[question.question_id] ?? "").trim().length > 0,
  );

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!complete || disabled || message.submitted) return;
    onSubmit({
      answers: message.questions.map((question) => ({
        question_id: question.question_id,
        answer: (answers[question.question_id] ?? "").trim(),
      })),
    });
  }

  return (
    <div className="chat-row assistant-row">
      <div className="assistant-avatar" aria-hidden="true">A</div>
      <div className="min-w-0 flex-1">
        <form onSubmit={submit} className="assessment-card">
          <div className="mb-4 flex items-start justify-between gap-3">
            <div>
              <div className="text-sm font-semibold text-zinc-900">Quick understanding check</div>
              <p className="mt-1 text-sm leading-6 text-zinc-500">
                {message.message || "Answer all three questions so I can decide what support you need next."}
              </p>
            </div>
            {message.submitted && <CheckCircle2 className="h-5 w-5 shrink-0 text-emerald-600" aria-label="Submitted" />}
          </div>

          <div className="grid gap-4">
            {message.questions.map((question, index) => (
              <label key={question.question_id} className="grid gap-2">
                <span className="text-sm font-medium leading-6 text-zinc-800">
                  {index + 1}. {question.question}
                </span>
                <textarea
                  rows={2}
                  value={answers[question.question_id] ?? ""}
                  onChange={(event) =>
                    setAnswers((current) => ({
                      ...current,
                      [question.question_id]: event.target.value,
                    }))
                  }
                  disabled={disabled || message.submitted}
                  maxLength={2000}
                  className="assessment-input"
                  placeholder="Your answer"
                />
              </label>
            ))}
          </div>

          {!message.submitted && (
            <button
              type="submit"
              disabled={disabled || !complete}
              className="mt-4 inline-flex items-center gap-2 rounded-xl bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-35"
            >
              Submit answers
              <ArrowUp className="h-4 w-4" aria-hidden="true" />
            </button>
          )}
        </form>
      </div>
    </div>
  );
}
