import type { ChatConversation } from "../types/chat";
import type { LearnerSetup } from "../types/tutor";

const LEGACY_LEARNER_KEY = "adaptmath.learnerSetup.v2";
const LEGACY_CHATS_KEY = "adaptmath.chats.v2";
const LEGACY_ACTIVE_CHAT_KEY = "adaptmath.activeChatId.v2";
const STUDENT_STORAGE_PREFIX = "adaptmath.student.v1";

type StudentStoragePart = "learnerSetup" | "chats" | "activeChatId";

function studentKey(studentId: string, part: StudentStoragePart): string {
  const normalizedStudentId = studentId.trim();
  if (!normalizedStudentId) {
    throw new Error("A student identity is required for browser storage.");
  }
  return `${STUDENT_STORAGE_PREFIX}.${encodeURIComponent(normalizedStudentId)}.${part}`;
}

function parseLearnerSetup(raw: string | null): LearnerSetup | null {
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

function parseChats(raw: string | null): ChatConversation[] {
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

/**
 * Move the pre-account-scoped browser data into the namespace of its known
 * owner. This is intentionally never migrated to the newly signed-in student
 * when the stored learner identity says it belongs to somebody else.
 */
export function migrateLegacyStudentStorage(studentId: string): boolean {
  const normalizedStudentId = studentId.trim();
  if (!normalizedStudentId) return false;

  const legacyLearnerRaw = window.localStorage.getItem(LEGACY_LEARNER_KEY);
  const legacyChatsRaw = window.localStorage.getItem(LEGACY_CHATS_KEY);
  const legacyActiveChatId = window.localStorage.getItem(LEGACY_ACTIVE_CHAT_KEY);
  if (!legacyLearnerRaw && !legacyChatsRaw && !legacyActiveChatId) return false;

  const legacyLearner = parseLearnerSetup(legacyLearnerRaw);
  if (legacyLearner && legacyLearner.studentId !== normalizedStudentId) {
    return false;
  }

  const learnerKey = studentKey(normalizedStudentId, "learnerSetup");
  const chatsKey = studentKey(normalizedStudentId, "chats");
  const activeChatKey = studentKey(normalizedStudentId, "activeChatId");

  if (legacyLearnerRaw && !window.localStorage.getItem(learnerKey)) {
    window.localStorage.setItem(learnerKey, legacyLearnerRaw);
  }
  if (legacyChatsRaw && !window.localStorage.getItem(chatsKey)) {
    window.localStorage.setItem(chatsKey, legacyChatsRaw);
  }
  if (legacyActiveChatId && !window.localStorage.getItem(activeChatKey)) {
    window.localStorage.setItem(activeChatKey, legacyActiveChatId);
  }

  window.localStorage.removeItem(LEGACY_LEARNER_KEY);
  window.localStorage.removeItem(LEGACY_CHATS_KEY);
  window.localStorage.removeItem(LEGACY_ACTIVE_CHAT_KEY);
  return true;
}

export function saveLearnerSetup(studentId: string, setup: LearnerSetup): void {
  window.localStorage.setItem(
    studentKey(studentId, "learnerSetup"),
    JSON.stringify({ ...setup, studentId }),
  );
}

export function loadLearnerSetup(studentId: string): LearnerSetup | null {
  const parsed = parseLearnerSetup(
    window.localStorage.getItem(studentKey(studentId, "learnerSetup")),
  );
  return parsed ? { ...parsed, studentId } : null;
}

export function clearLearnerSetup(studentId: string): void {
  window.localStorage.removeItem(studentKey(studentId, "learnerSetup"));
}

export function saveChats(studentId: string, chats: ChatConversation[]): void {
  window.localStorage.setItem(
    studentKey(studentId, "chats"),
    JSON.stringify(chats),
  );
}

export function loadChats(studentId: string): ChatConversation[] {
  return parseChats(
    window.localStorage.getItem(studentKey(studentId, "chats")),
  );
}

export function saveActiveChatId(
  studentId: string,
  chatId: string | null,
): void {
  const key = studentKey(studentId, "activeChatId");
  if (chatId) {
    window.localStorage.setItem(key, chatId);
  } else {
    window.localStorage.removeItem(key);
  }
}

export function loadActiveChatId(studentId: string): string | null {
  return window.localStorage.getItem(studentKey(studentId, "activeChatId"));
}
