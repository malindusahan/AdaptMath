import { useState, type FormEvent } from "react";
import { ArrowRight, Sparkles } from "lucide-react";
import type { AuthenticatedUser } from "../types/auth";
import type { LearnerSetup } from "../types/tutor";

interface LearnerSetupFormProps {
  initialValue?: LearnerSetup | null;
  identity: AuthenticatedUser;
  onContinue: (setup: LearnerSetup) => void;
}

export function LearnerSetupForm({ identity, initialValue, onContinue }: LearnerSetupFormProps) {
  const accountAge = identity.age && identity.age >= 8 && identity.age <= 18
    ? identity.age
    : 15;
  const [age, setAge] = useState(initialValue?.age ?? accountAge);
  const [topic, setTopic] = useState(initialValue?.topic ?? "Algebra");
  const [subtopic, setSubtopic] = useState(initialValue?.subtopic ?? "Linear equations");
  const [targetSkill, setTargetSkill] = useState(
    initialValue?.targetSkill ?? "",
  );

  const invalid =
    !topic.trim() ||
    age < 8 ||
    age > 18;

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (invalid) return;
    onContinue({
      studentId: identity.student_id,
      age,
      topic: topic.trim(),
      subtopic: subtopic.trim(),
      targetSkill: targetSkill.trim(),
    });
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-white px-4 py-10">
      <div className="w-full max-w-md">
        <div className="mb-8 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-zinc-900 text-white">
            <Sparkles className="h-5 w-5" aria-hidden="true" />
          </div>
          <h1 className="mt-5 text-2xl font-semibold tracking-tight text-zinc-950">Welcome to AdaptMath</h1>
          <p className="mt-2 text-sm leading-6 text-zinc-500">Signed in as <span className="font-medium text-zinc-700">{identity.username}</span>. Set the lesson context, then start learning.</p>
          {import.meta.env.DEV && <p className="mt-1 text-[11px] text-zinc-400">Resolved student: {identity.student_id}</p>}
        </div>

        <form onSubmit={submit} className="grid gap-4 rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm">
          <label className="grid gap-1.5 text-sm font-medium text-zinc-700">
            Age
            <input className="chat-field" type="number" min={8} max={18} value={age} onChange={(event) => setAge(Number(event.target.value))} />
          </label>
          <label className="grid gap-1.5 text-sm font-medium text-zinc-700">
            Topic
            <input className="chat-field" value={topic} maxLength={128} onChange={(event) => setTopic(event.target.value)} />
          </label>
          <label className="grid gap-1.5 text-sm font-medium text-zinc-700">
            Subtopic <span className="font-normal text-zinc-400">(optional)</span>
            <input className="chat-field" value={subtopic} maxLength={128} onChange={(event) => setSubtopic(event.target.value)} />
          </label>
          <label className="grid gap-1.5 text-sm font-medium text-zinc-700">
            Canonical BKT skill override <span className="font-normal text-zinc-400">(optional)</span>
            <input className="chat-field" value={targetSkill} maxLength={128} placeholder="Automatically identified from each question" onChange={(event) => setTargetSkill(event.target.value)} />
          </label>
          <button type="submit" disabled={invalid} className="mt-2 flex items-center justify-center gap-2 rounded-xl bg-zinc-900 px-4 py-3 text-sm font-medium text-white hover:bg-zinc-800 disabled:opacity-40">
            Continue
            <ArrowRight className="h-4 w-4" aria-hidden="true" />
          </button>
          <p className="text-center text-[11px] leading-4 text-zinc-400">Leave the skill override blank to use Memory's automatic topic classifier.</p>
        </form>
      </div>
    </main>
  );
}
