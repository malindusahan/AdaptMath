import { useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ArrowUp, LoaderCircle } from "lucide-react";

interface ChatComposerProps {
  disabled?: boolean;
  busy?: boolean;
  placeholder?: string;
  onSubmit: (question: string) => void;
}

export function ChatComposer({
  disabled = false,
  busy = false,
  placeholder = "Ask a maths question",
  onSubmit,
}: ChatComposerProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  function send() {
    const cleaned = value.trim();
    if (!cleaned || disabled || busy) return;
    onSubmit(cleaned);
    setValue("");
    requestAnimationFrame(() => textareaRef.current?.focus());
  }

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    send();
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  }

  return (
    <form onSubmit={submit} className="mx-auto w-full max-w-3xl">
      <div className={`chat-composer ${disabled ? "opacity-70" : ""}`}>
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(event) => setValue(event.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled || busy}
          maxLength={8000}
          placeholder={placeholder}
          aria-label="Ask AdaptMath"
          className="max-h-44 min-h-12 flex-1 resize-none bg-transparent px-2 py-3 text-[15px] leading-6 text-slate-900 outline-none placeholder:text-slate-400 disabled:cursor-not-allowed dark:text-slate-100 dark:placeholder:text-slate-500"
        />
        <button
          type="submit"
          disabled={!value.trim() || disabled || busy}
          className="mb-1 flex h-10 w-10 shrink-0 items-center justify-center self-end rounded-full bg-gradient-to-br from-indigo-600 to-violet-600 text-white shadow-md shadow-indigo-500/20 transition hover:scale-105 disabled:scale-100 disabled:bg-none disabled:bg-slate-200 disabled:text-slate-400 disabled:shadow-none dark:disabled:bg-slate-700 dark:disabled:text-slate-500"
          aria-label="Send question"
        >
          {busy ? (
            <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden="true" />
          ) : (
            <ArrowUp className="h-4 w-4" aria-hidden="true" />
          )}
        </button>
      </div>
      <p className="mt-2 text-center text-[11px] leading-4 text-slate-400 dark:text-slate-500">
        AdaptMath can make mistakes. Check important mathematical results.
      </p>
    </form>
  );
}
