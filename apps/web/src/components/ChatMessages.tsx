import { AlertTriangle, CheckCircle2, RefreshCw } from "lucide-react";
import type { ChatMessage } from "../types/chat";
import type { TutorAnswerPayload } from "../types/tutor";
import { ChatAssessment } from "./ChatAssessment";
import { MathMarkdown } from "./MathMarkdown";

interface ChatMessagesProps {
  messages: ChatMessage[];
  busy?: boolean;
  onSubmitAssessment: (messageId: string, payload: TutorAnswerPayload) => void;
  onResume: () => void;
}

export function ChatMessages({
  messages,
  busy = false,
  onSubmitAssessment,
  onResume,
}: ChatMessagesProps) {
  return (
    <div className="mx-auto w-full max-w-3xl pb-8 pt-4 sm:pt-8">
      {messages.map((message) => {
        if (message.type === "user") {
          return (
            <div key={message.id} className="chat-row user-row">
              <div className="user-bubble">{message.content}</div>
            </div>
          );
        }

        if (message.type === "assistant") {
          return (
            <div key={message.id} className="chat-row assistant-row">
              <div className="assistant-avatar" aria-hidden="true">A</div>
              <div className="min-w-0 flex-1 pt-0.5">
                {message.reteachRound > 0 && (
                  <div className="mb-2 text-xs font-bold text-violet-600 dark:text-violet-300">
                    Adapted explanation · round {message.reteachRound}
                  </div>
                )}
                <MathMarkdown>{message.content}</MathMarkdown>
              </div>
            </div>
          );
        }

        if (message.type === "assessment") {
          return (
            <ChatAssessment
              key={message.id}
              message={message}
              disabled={busy}
              onSubmit={(payload) => onSubmitAssessment(message.id, payload)}
            />
          );
        }

        if (message.type === "evaluation") {
          return (
            <div key={message.id} className="chat-row assistant-row">
              <div className="assistant-avatar" aria-hidden="true">A</div>
              <div className="feedback-card">
                <div className="flex items-start gap-3">
                  {message.evaluation.needs_reteaching ? (
                    <RefreshCw className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" aria-hidden="true" />
                  ) : (
                    <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" aria-hidden="true" />
                  )}
                  <div>
                    <div className="text-sm font-bold text-slate-900 dark:text-slate-100">
                      {message.evaluation.needs_reteaching ? "I’ll adjust the explanation" : "Understanding confirmed"}
                    </div>
                    {message.evaluation.overall_feedback && (
                      <p className="mt-1 text-sm leading-6 text-slate-600 dark:text-slate-300">{message.evaluation.overall_feedback}</p>
                    )}
                  </div>
                </div>
                {message.evaluation.identified_errors.length > 0 && (
                  <div className="mt-3 border-t border-slate-200 pt-3 dark:border-slate-700">
                    <div className="text-xs font-bold uppercase tracking-wide text-slate-500 dark:text-slate-400">What to work on</div>
                    <ul className="mt-2 grid gap-1 text-sm leading-6 text-slate-700 dark:text-slate-300">
                      {message.evaluation.identified_errors.map((error) => (
                        <li key={error}>• {error}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            </div>
          );
        }

        if (message.type === "recovery") {
          return (
            <div key={message.id} className="chat-row assistant-row">
              <div className="assistant-avatar" aria-hidden="true">A</div>
              <div className="feedback-card border-amber-200 bg-amber-50/70 dark:border-amber-500/30 dark:bg-amber-500/10">
                <div className="flex gap-3">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" aria-hidden="true" />
                  <div>
                    <div className="text-sm font-bold text-slate-900 dark:text-slate-100">Your lesson is safe</div>
                    <p className="mt-1 text-sm leading-6 text-slate-600 dark:text-slate-300">{message.content}</p>
                    <button
                      type="button"
                      onClick={onResume}
                      disabled={busy}
                      className="mt-3 rounded-xl bg-indigo-600 px-3.5 py-2 text-sm font-bold text-white hover:bg-indigo-500 disabled:opacity-40"
                    >
                      Resume lesson
                    </button>
                  </div>
                </div>
              </div>
            </div>
          );
        }

        return (
          <div key={message.id} className="chat-row assistant-row">
            <div className="assistant-avatar" aria-hidden="true">A</div>
            <div className="flex items-center gap-2 rounded-xl bg-emerald-50 px-4 py-3 text-sm text-emerald-800 dark:bg-emerald-500/10 dark:text-emerald-300">
              <CheckCircle2 className="h-4 w-4" aria-hidden="true" />
              {message.content}
            </div>
          </div>
        );
      })}
    </div>
  );
}
