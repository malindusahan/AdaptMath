import type { ChatConversation } from "../types/chat";
import type { LearnerSetup } from "../types/tutor";

const LEARNER_KEY = "adaptmath.learnerSetup.v2";
const CHATS_KEY = "adaptmath.chats.v2";
const ACTIVE_CHAT_KEY = "adaptmath.activeChatId.v2";

export function saveLearnerSetup(setup: LearnerSetup): void {
  window.localStorage.setItem(LEARNER_KEY, JSON.stringify(setup));
}

export function loadLearnerSetup(): LearnerSetup | null {
  const raw = window.localStorage.getItem(LEARNER_KEY);
  if (!raw) return null;

  try {
    const parsed = JSON.parse(raw) as LearnerSetup;
    if (
      typeof parsed.studentId === "string" &&
      typeof parsed.age === "number" &&
      typeof parsed.topic === "string" &&
      typeof parsed.subtopic === "string" &&
      typeof parsed.targetSkill === "string"
    ) {
      return parsed;
    }
  } catch {
    // Ignore malformed browser storage.
  }

  return null;
}

export function clearLearnerSetup(): void {
  window.localStorage.removeItem(LEARNER_KEY);
}

export function saveChats(chats: ChatConversation[]): void {
  window.localStorage.setItem(CHATS_KEY, JSON.stringify(chats));
}

export function loadChats(): ChatConversation[] {
  const raw = window.localStorage.getItem(CHATS_KEY);
  if (!raw) return [];

  try {
    const parsed = JSON.parse(raw) as ChatConversation[];
    if (!Array.isArray(parsed)) return [];

    return parsed.filter(
      (chat) =>
        chat &&
        typeof chat.id === "string" &&
        typeof chat.title === "string" &&
        Array.isArray(chat.messages),
    );
  } catch {
    return [];
  }
}

export function saveActiveChatId(chatId: string | null): void {
  if (chatId) {
    window.localStorage.setItem(ACTIVE_CHAT_KEY, chatId);
  } else {
    window.localStorage.removeItem(ACTIVE_CHAT_KEY);
  }
}

export function loadActiveChatId(): string | null {
  return window.localStorage.getItem(ACTIVE_CHAT_KEY);
}
