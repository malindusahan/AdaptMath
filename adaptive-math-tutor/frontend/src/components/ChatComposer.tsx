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
          className="max-h-44 min-h-12 flex-1 resize-none bg-transparent px-2 py-3 text-[15px] leading-6 text-zinc-900 outline-none placeholder:text-zinc-400 disabled:cursor-not-allowed"
        />
        <button
          type="submit"
          disabled={!value.trim() || disabled || busy}
          className="mb-1 flex h-9 w-9 shrink-0 items-center justify-center self-end rounded-full bg-zinc-900 text-white transition hover:bg-zinc-700 disabled:bg-zinc-200 disabled:text-zinc-400"
          aria-label="Send question"
        >
          {busy ? (
            <LoaderCircle className="h-4 w-4 animate-spin" aria-hidden="true" />
          ) : (
            <ArrowUp className="h-4 w-4" aria-hidden="true" />
          )}
        </button>
      </div>
      <p className="mt-2 text-center text-[11px] leading-4 text-zinc-400">
        AdaptMath can make mistakes. Check important mathematical results.
      </p>
    </form>
  );
}
