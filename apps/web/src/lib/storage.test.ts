import { beforeEach, describe, expect, it } from "vitest";

import type { ChatConversation } from "../types/chat";
import type { LearnerSetup } from "../types/tutor";
import {
  loadActiveChatId,
  loadChats,
  loadLearnerSetup,
  migrateLegacyStudentStorage,
  saveActiveChatId,
  saveChats,
  saveLearnerSetup,
} from "./storage";

class MemoryStorage implements Storage {
  private readonly values = new Map<string, string>();

  get length(): number {
    return this.values.size;
  }

  clear(): void {
    this.values.clear();
  }

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  key(index: number): string | null {
    return [...this.values.keys()][index] ?? null;
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

function chat(id: string, title: string): ChatConversation {
  return {
    id,
    title,
    createdAt: "2026-10-16T00:00:00.000Z",
    updatedAt: "2026-10-16T00:00:00.000Z",
    messages: [],
    activeThreadId: null,
    session: null,
  };
}

function learner(studentId: string): LearnerSetup {
  return {
    studentId,
    age: 14,
    topic: "Mathematics",
    subtopic: "",
    targetSkill: "",
    targetSkillSource: null,
  };
}

beforeEach(() => {
  Object.defineProperty(globalThis, "window", {
    configurable: true,
    value: { localStorage: new MemoryStorage() },
  });
});

describe("student-scoped browser storage", () => {
  it("keeps chats, learner settings, and the active chat separate by student", () => {
    saveChats("student-a", [chat("chat-a", "Student A question")]);
    saveActiveChatId("student-a", "chat-a");
    saveLearnerSetup("student-a", learner("student-a"));

    saveChats("student-b", [chat("chat-b", "Student B question")]);
    saveActiveChatId("student-b", "chat-b");
    saveLearnerSetup("student-b", learner("student-b"));

    expect(loadChats("student-a").map((item) => item.id)).toEqual(["chat-a"]);
    expect(loadActiveChatId("student-a")).toBe("chat-a");
    expect(loadLearnerSetup("student-a")?.studentId).toBe("student-a");

    expect(loadChats("student-b").map((item) => item.id)).toEqual(["chat-b"]);
    expect(loadActiveChatId("student-b")).toBe("chat-b");
    expect(loadLearnerSetup("student-b")?.studentId).toBe("student-b");
  });

  it("migrates legacy chats only into the matching student's namespace", () => {
    window.localStorage.setItem(
      "adaptmath.learnerSetup.v2",
      JSON.stringify(learner("student-a")),
    );
    window.localStorage.setItem(
      "adaptmath.chats.v2",
      JSON.stringify([chat("legacy-a", "Legacy question")]),
    );
    window.localStorage.setItem("adaptmath.activeChatId.v2", "legacy-a");

    expect(migrateLegacyStudentStorage("student-b")).toBe(false);
    expect(loadChats("student-b")).toEqual([]);
    expect(window.localStorage.getItem("adaptmath.chats.v2")).not.toBeNull();

    expect(migrateLegacyStudentStorage("student-a")).toBe(true);
    expect(loadChats("student-a").map((item) => item.id)).toEqual(["legacy-a"]);
    expect(loadActiveChatId("student-a")).toBe("legacy-a");
    expect(window.localStorage.getItem("adaptmath.chats.v2")).toBeNull();
  });
});
