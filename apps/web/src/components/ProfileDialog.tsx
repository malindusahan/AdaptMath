import { useEffect, useState, type FormEvent } from "react";
import { X } from "lucide-react";
import type { AuthenticatedUser } from "../types/auth";
import type { LearnerSetup } from "../types/tutor";

interface ProfileDialogProps {
  open: boolean;
  identity: AuthenticatedUser;
  learner: LearnerSetup;
  onClose: () => void;
  onSave: (learner: LearnerSetup) => void;
}

export function ProfileDialog({ open, identity, learner, onClose, onSave }: ProfileDialogProps) {
  const [draft, setDraft] = useState(learner);

  useEffect(() => {
    if (open) setDraft(learner);
  }, [learner, open]);

  if (!open) return null;

  const invalid =
    !draft.topic.trim() ||
    draft.age < 8 ||
    draft.age > 18;

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (invalid) return;
    onSave({
      ...draft,
      studentId: identity.student_id,
      topic: draft.topic.trim(),
      subtopic: draft.subtopic.trim(),
      targetSkill: draft.targetSkill.trim(),
    });
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/35 p-4" role="presentation">
      <div className="w-full max-w-lg rounded-2xl border border-zinc-200 bg-white shadow-2xl" role="dialog" aria-modal="true" aria-labelledby="profile-title">
        <div className="flex items-center justify-between border-b border-zinc-100 px-5 py-4">
          <div>
            <h2 id="profile-title" className="font-semibold text-zinc-950">Learner context</h2>
            <p className="mt-0.5 text-xs text-zinc-500">Signed in as {identity.username}. Account identity is read-only.</p>
          </div>
          <button type="button" onClick={onClose} className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-100" aria-label="Close settings">
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>

        <form onSubmit={submit} className="grid gap-4 p-5">
          {import.meta.env.DEV && (
            <div className="rounded-xl bg-zinc-50 px-3.5 py-2.5 text-xs text-zinc-500">
              Resolved student: <span className="font-mono text-zinc-700">{identity.student_id}</span>
            </div>
          )}
          <label className="grid gap-1.5 text-sm font-medium text-zinc-700">
            Age
            <input
              className="chat-field"
              type="number"
              min={8}
              max={18}
              value={draft.age}
              onChange={(event) => setDraft((current) => ({ ...current, age: Number(event.target.value) }))}
            />
          </label>
          <label className="grid gap-1.5 text-sm font-medium text-zinc-700">
            Topic
            <input
              className="chat-field"
              value={draft.topic}
              maxLength={128}
              onChange={(event) => setDraft((current) => ({ ...current, topic: event.target.value }))}
            />
          </label>
          <label className="grid gap-1.5 text-sm font-medium text-zinc-700">
            Subtopic <span className="font-normal text-zinc-400">(optional)</span>
            <input
              className="chat-field"
              value={draft.subtopic}
              maxLength={128}
              onChange={(event) => setDraft((current) => ({ ...current, subtopic: event.target.value }))}
            />
          </label>
          <label className="grid gap-1.5 text-sm font-medium text-zinc-700">
            Canonical BKT skill override <span className="font-normal text-zinc-400">(optional)</span>
            <input
              className="chat-field"
              value={draft.targetSkill}
              maxLength={128}
              placeholder="Automatically identified from each question"
              onChange={(event) => setDraft((current) => ({ ...current, targetSkill: event.target.value }))}
            />
          </label>

          <div className="mt-2 flex justify-end gap-2">
            <button type="button" onClick={onClose} className="rounded-xl px-4 py-2.5 text-sm font-medium text-zinc-600 hover:bg-zinc-100">Cancel</button>
            <button type="submit" disabled={invalid} className="rounded-xl bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40">Save</button>
          </div>
        </form>
      </div>
    </div>
  );
}
