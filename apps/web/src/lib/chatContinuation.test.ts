import { describe, expect, it } from "vitest";
import type { ChatMessage } from "../types/chat";
import {
  continuationHistory,
  messagesForPausedLesson,
  originalLessonQuestion,
} from "./chatContinuation";

const messages: ChatMessage[] = [
  {
    id: "question",
    type: "user",
    createdAt: "2026-01-01T00:00:00.000Z",
    content: "Solve 3x + 5 = 20.",
  },
  {
    id: "teacher",
    type: "assistant",
    createdAt: "2026-01-01T00:00:01.000Z",
    content: "What should we undo first?",
    threadId: "old-thread",
    reteachRound: 0,
    turnCount: 1,
  },
  {
    id: "reply",
    type: "user",
    createdAt: "2026-01-01T00:00:02.000Z",
    content: "Subtract 5.",
  },
];

describe("chat continuation", () => {
  it("keeps the original problem separate from the resumed dialogue", () => {
    expect(originalLessonQuestion(messages, "fallback")).toBe("Solve 3x + 5 = 20.");
    expect(continuationHistory(messages)).toEqual([
      { role: "teacher", content: "What should we undo first?" },
      { role: "student", content: "Subtract 5." },
    ]);
  });

  it("removes only controls that cannot be resumed after the old attempt closes", () => {
    const pausedMessages: ChatMessage[] = [
      ...messages,
      {
        id: "assessment",
        type: "assessment",
        createdAt: "2026-01-01T00:00:03.000Z",
        threadId: "old-thread",
        reteachRound: 0,
        questions: [
          { question_id: "q1", question: "What is x?" },
          { question_id: "q2", question: "Check x." },
          { question_id: "q3", question: "Explain why." },
        ],
        submitted: false,
      },
      {
        id: "recovery",
        type: "recovery",
        createdAt: "2026-01-01T00:00:04.000Z",
        threadId: "old-thread",
        content: "Resume",
      },
    ];

    expect(messagesForPausedLesson(pausedMessages)).toEqual(messages);
  });
});
