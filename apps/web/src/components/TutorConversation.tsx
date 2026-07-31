import { Bot, UserRound } from "lucide-react";
import { MathMarkdown } from "./MathMarkdown";

interface TutorConversationProps {
  question: string;
  tutorResponse?: string | null;
  reteachRound: number;
}

export function TutorConversation({
  question,
  tutorResponse,
  reteachRound,
}: TutorConversationProps) {
  return (
    <div className="grid gap-5">
      {question && (
        <div className="flex justify-end gap-3">
          <div className="max-w-[88%] rounded-3xl rounded-tr-lg bg-slate-950 px-5 py-4 text-sm leading-7 text-white shadow-sm sm:max-w-[78%]">
            {question}
          </div>
          <div className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-slate-200 text-slate-700">
            <UserRound className="h-4 w-4" aria-hidden="true" />
          </div>
        </div>
      )}

      {tutorResponse && (
        <div className="flex items-start gap-3">
          <div className="mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-indigo-600 text-white shadow-sm shadow-indigo-200">
            <Bot className="h-5 w-5" aria-hidden="true" />
          </div>
          <div className="min-w-0 max-w-[92%] rounded-3xl rounded-tl-lg border border-slate-200 bg-white px-5 py-5 shadow-sm sm:px-6">
            {reteachRound > 0 && (
              <div className="mb-4 inline-flex rounded-full bg-indigo-50 px-3 py-1 text-xs font-bold text-indigo-700">
                Adapted explanation · round {reteachRound}
              </div>
            )}
            <MathMarkdown>{tutorResponse}</MathMarkdown>
          </div>
        </div>
      )}
    </div>
  );
}
