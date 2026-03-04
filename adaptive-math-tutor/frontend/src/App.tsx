import { useEffect, useMemo, useRef, useState } from "react";
import { Menu, Plus, Settings2 } from "lucide-react";
import {
  abandonTutorSession,
  ApiError,
  getActiveTutorSession,
  getHealth,
  getReady,
  getTutorSession,
  resumeTutorSession,
  startRecommendedPractice,
  startTutorSession,
  submitTutorAnswers,
  submitTutorTurn,
} from "./api/client";
import {
  AUTH_INVALID_EVENT,
  clearStudentHint,
  getStoredAuthToken,
  getStoredStudentHint,
  logout as logoutUser,
  restoreAuthentication,
  saveStudentHint,
} from "./api/auth";
import { AuthScreen } from "./components/AuthScreen";
import { ChatComposer } from "./components/ChatComposer";
import { ChatMessages } from "./components/ChatMessages";
import { ChatSidebar } from "./components/ChatSidebar";
import { EmptyChat } from "./components/EmptyChat";
import { ProfileDialog } from "./components/ProfileDialog";
import { StudentProfilePage } from "./components/StudentProfilePage";
import {
  clearLearnerSetup,
  loadActiveChatId,
  loadChats,
  loadLearnerSetup,
  saveActiveChatId,
  saveChats,
  saveLearnerSetup,
} from "./lib/storage";
import type { ChatConversation, ChatMessage } from "./types/chat";
import type { AuthenticatedUser } from "./types/auth";
import type {
  LearnerSetup,
  TutorAnswerPayload,
  TutorSessionResponse,
} from "./types/tutor";

type BackendStatus = "checking" | "ready" | "offline";
type AuthStatus = "checking" | "authenticated" | "unauthenticated";
type AppPage = "tutor" | "profile";

function pageFromPath(): AppPage {
  return window.location.pathname === "/profile" ? "profile" : "tutor";
}

function nowIso(): string {
  return new Date().toISOString();
}

function createEmptyChat(): ChatConversation {
  const now = nowIso();
  return {
    id: crypto.randomUUID(),
    title: "New chat",
    createdAt: now,
    updatedAt: now,
    messages: [],
    activeThreadId: null,
    session: null,
  };
}

function learnerSetupForAccount(
  user: AuthenticatedUser,
  existing: LearnerSetup | null,
): LearnerSetup {
  if (existing) return { ...existing, studentId: user.student_id };

  const accountAge = user.age && user.age >= 8 && user.age <= 18 ? user.age : 15;
  return {
    studentId: user.student_id,
    age: accountAge,
    topic: "Mathematics",
    subtopic: "",
    targetSkill: "",
  };
}

function titleFromQuestion(question: string): string {
  const clean = question.replace(/\s+/g, " ").trim();
  return clean.length <= 42 ? clean : `${clean.slice(0, 39).trim()}…`;
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 409) {
      if (error.detail.includes("currently active")) {
        return "Finish the active lesson before starting another one.";
      }
      if (error.detail.includes("interrupted by a server restart")) {
        return "This tutoring session was interrupted by a server restart and cannot be safely continued. Start a new session.";
      }
    }
    if (error.status === 503) {
      return "The AI tutoring service is temporarily unavailable or rate-limited. Your saved lesson can be resumed when the service is available.";
    }
    if (error.status === 504) {
      return "The AI tutoring service took too long to respond. Your saved lesson can be resumed safely.";
    }
    return error.detail;
  }
  if (error instanceof Error) return error.message;
  return "Something went wrong while contacting AdaptMath.";
}

function hasAssistantSnapshot(
  messages: ChatMessage[],
  threadId: string,
  turnCount: number,
): boolean {
  return messages.some(
    (message) =>
      message.type === "assistant" &&
      message.threadId === threadId &&
      message.turnCount === turnCount,
  );
}

function hasEvaluationSnapshot(
  messages: ChatMessage[],
  threadId: string,
  reteachRound: number,
): boolean {
  return messages.some(
    (message) =>
      message.type === "evaluation" &&
      message.threadId === threadId &&
      message.reteachRound === reteachRound,
  );
}

function hasAssessmentSnapshot(
  messages: ChatMessage[],
  threadId: string,
  reteachRound: number,
): boolean {
  return messages.some(
    (message) =>
      message.type === "assessment" &&
      message.threadId === threadId &&
      message.reteachRound === reteachRound,
  );
}

function hasRecoverySnapshot(messages: ChatMessage[], threadId: string): boolean {
  return messages.some(
    (message) => message.type === "recovery" && message.threadId === threadId,
  );
}

function hasCompletionSnapshot(messages: ChatMessage[], threadId: string): boolean {
  return messages.some(
    (message) => message.type === "completion" && message.threadId === threadId,
  );
}

function reconcileSession(
  chat: ChatConversation,
  session: TutorSessionResponse,
): ChatConversation {
  const messages = [...chat.messages];
  const timestamp = nowIso();

  if (
    session.tutor_response &&
    session.turn_count > 0 &&
    !hasAssistantSnapshot(messages, session.thread_id, session.turn_count)
  ) {
    messages.push({
      id: crypto.randomUUID(),
      type: "assistant",
      createdAt: timestamp,
      content: session.tutor_response,
      threadId: session.thread_id,
      reteachRound: session.reteach_round,
      turnCount: session.turn_count,
    });
  }

  if (
    session.evaluation &&
    !hasEvaluationSnapshot(messages, session.thread_id, session.reteach_round)
  ) {
    messages.push({
      id: crypto.randomUUID(),
      type: "evaluation",
      createdAt: timestamp,
      threadId: session.thread_id,
      reteachRound: session.reteach_round,
      evaluation: session.evaluation,
    });
  }

  if (
    session.status === "assessment_required" &&
    session.questions.length === 3 &&
    !hasAssessmentSnapshot(messages, session.thread_id, session.reteach_round)
  ) {
    messages.push({
      id: crypto.randomUUID(),
      type: "assessment",
      createdAt: timestamp,
      threadId: session.thread_id,
      reteachRound: session.reteach_round,
      message: session.assessment_message,
      questions: session.questions,
      submitted: false,
    });
  }

  if (
    session.status === "recovery_required" &&
    !hasRecoverySnapshot(messages, session.thread_id)
  ) {
    messages.push({
      id: crypto.randomUUID(),
      type: "recovery",
      createdAt: timestamp,
      threadId: session.thread_id,
      content:
        "A backend step paused, but your progress was saved. Resume from the last durable checkpoint.",
    });
  }

  if (
    session.status === "complete" &&
    session.route !== "gemini_general_chat" &&
    !hasCompletionSnapshot(messages, session.thread_id)
  ) {
    messages.push({
      id: crypto.randomUUID(),
      type: "completion",
      createdAt: timestamp,
      threadId: session.thread_id,
      content: "This lesson is complete. You can keep chatting and ask another maths question below.",
    });
  }

  return {
    ...chat,
    messages,
    session,
    activeThreadId: session.status === "complete" ? null : session.thread_id,
    updatedAt: timestamp,
  };
}

export default function App() {
  const [authStatus, setAuthStatus] = useState<AuthStatus>("checking");
  const [authUser, setAuthUser] = useState<AuthenticatedUser | null>(null);
  const [authNotice, setAuthNotice] = useState<string | null>(null);
  const [learner, setLearner] = useState<LearnerSetup | null>(null);
  const [chats, setChats] = useState<ChatConversation[]>(() => loadChats());
  const [activeChatId, setActiveChatId] = useState<string | null>(() => loadActiveChatId());
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [page, setPage] = useState<AppPage>(() => pageFromPath());
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const startRequestInFlightRef = useRef(false);

  const activeChat = useMemo(
    () => chats.find((chat) => chat.id === activeChatId) ?? null,
    [activeChatId, chats],
  );

  function activateAuthenticatedUser(user: AuthenticatedUser) {
    const storedSetup = loadLearnerSetup();
    const previousStudentId = getStoredStudentHint() ?? storedSetup?.studentId ?? null;
    const identityChanged = Boolean(
      previousStudentId && previousStudentId !== user.student_id,
    );

    if (identityChanged) {
      clearLearnerSetup();
      saveChats([]);
      saveActiveChatId(null);
      setChats([]);
      setActiveChatId(null);
    }

    const resolvedSetup = learnerSetupForAccount(
      user,
      identityChanged ? null : storedSetup,
    );
    saveLearnerSetup(resolvedSetup);
    setLearner(resolvedSetup);

    saveStudentHint(user.student_id);
    setAuthUser(user);
    setAuthStatus("authenticated");
    setAuthNotice(null);
    setError(null);
  }

  useEffect(() => {
    function followBrowserHistory() {
      setPage(pageFromPath());
      setSettingsOpen(false);
    }
    window.addEventListener("popstate", followBrowserHistory);
    return () => window.removeEventListener("popstate", followBrowserHistory);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const hadStoredToken = Boolean(getStoredAuthToken());

    async function restore() {
      const user = await restoreAuthentication();
      if (cancelled) return;
      if (user) {
        activateAuthenticatedUser(user);
      } else {
        setAuthStatus("unauthenticated");
        if (hadStoredToken) {
          setAuthNotice("Your saved session is no longer valid. Please sign in again.");
        }
      }
    }

    function authenticationInvalidated() {
      if (cancelled) return;
      setAuthUser(null);
      setLearner(null);
      setAuthStatus("unauthenticated");
      setAuthNotice("Your authentication session ended. Please sign in again.");
    }

    window.addEventListener(AUTH_INVALID_EVENT, authenticationInvalidated);
    void restore();
    return () => {
      cancelled = true;
      window.removeEventListener(AUTH_INVALID_EVENT, authenticationInvalidated);
    };
  }, []);

  useEffect(() => {
    if (!learner) return;

    if (chats.length === 0) {
      const chat = createEmptyChat();
      setChats([chat]);
      setActiveChatId(chat.id);
      saveActiveChatId(chat.id);
      return;
    }

    if (!activeChatId || !chats.some((chat) => chat.id === activeChatId)) {
      const fallback = [...chats].sort(
        (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
      )[0];
      setActiveChatId(fallback.id);
      saveActiveChatId(fallback.id);
    }
  }, [activeChatId, chats, learner]);

  useEffect(() => {
    saveChats(chats);
  }, [chats]);

  useEffect(() => {
    let cancelled = false;

    async function checkBackend() {
      const [alive, ready] = await Promise.all([getHealth(), getReady()]);
      if (!cancelled) setBackendStatus(alive && ready ? "ready" : "offline");
    }

    void checkBackend();
    const timer = window.setInterval(checkBackend, 30_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const studentId = learner?.studentId;
    if (!studentId) return;

    let cancelled = false;
    async function recoverServerLesson() {
      try {
        const session = await getActiveTutorSession();
        if (!cancelled && session) attachActiveSession(session);
      } catch (caught) {
        if (!cancelled && (!(caught instanceof ApiError) || caught.status !== 404)) {
          setError(errorMessage(caught));
        }
      }
    }

    void recoverServerLesson();
    return () => {
      cancelled = true;
    };
  }, [learner?.studentId]);

  useEffect(() => {
    const chatId = activeChat?.id;
    const threadId = activeChat?.activeThreadId;

    if (typeof chatId !== "string" || typeof threadId !== "string") return;

    let cancelled = false;

    async function recover(currentChatId: string, persistedThreadId: string) {
      try {
        const session = await getTutorSession(persistedThreadId);
        if (cancelled) return;
        setChats((current) =>
          current.map((chat) =>
            chat.id === currentChatId ? reconcileSession(chat, session) : chat,
          ),
        );
      } catch (caught) {
        if (!cancelled && !(caught instanceof ApiError && caught.status === 404)) {
          setError(errorMessage(caught));
        }
      }
    }

    void recover(chatId, threadId);
    return () => {
      cancelled = true;
    };
  }, [activeChat?.activeThreadId, activeChat?.id]);

  useEffect(() => {
    if (
      import.meta.env.DEV &&
      activeChat?.session?.status === "pedagogical_move_required" &&
      activeChat.activeThreadId
    ) {
      console.info(
        `[AdaptMath dev] Awaiting external Omash move for thread ${activeChat.activeThreadId}`,
      );
    }
  }, [activeChat?.activeThreadId, activeChat?.session?.status]);

  // Omash owns pedagogical-move selection. While the backend is waiting for
  // that external component, the learner sees only a neutral preparation state.
  // Polling lets the UI pick up the next teacher turn after Omash supplies it.
  useEffect(() => {
    const chatId = activeChat?.id;
    const threadId = activeChat?.activeThreadId;
    const status = activeChat?.session?.status;

    if (
      typeof chatId !== "string" ||
      typeof threadId !== "string" ||
      status !== "pedagogical_move_required"
    ) {
      return;
    }

    const currentChatId = chatId;
    const persistedThreadId = threadId;
    let cancelled = false;
    let inFlight = false;

    async function pollForTeacherTurn() {
      if (inFlight) return;
      inFlight = true;
      try {
        const latest = await getTutorSession(persistedThreadId);
        if (!cancelled && latest.status !== "pedagogical_move_required") {
          setChats((current) =>
            current.map((chat) =>
              chat.id === currentChatId ? reconcileSession(chat, latest) : chat,
            ),
          );
        }
      } catch (caught) {
        if (!cancelled && !(caught instanceof ApiError && caught.status === 404)) {
          setError(errorMessage(caught));
        }
      } finally {
        inFlight = false;
      }
    }

    void pollForTeacherTurn();
    const timer = window.setInterval(pollForTeacherTurn, 1_500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [
    activeChat?.activeThreadId,
    activeChat?.id,
    activeChat?.session?.status,
  ]);

  useEffect(() => {
    requestAnimationFrame(() => {
      const scroller = scrollerRef.current;
      if (scroller) scroller.scrollTop = scroller.scrollHeight;
    });
  }, [activeChat?.messages.length, activeChat?.session?.status, busy]);

  async function endActiveLesson(chat: ChatConversation): Promise<boolean> {
    const threadId = chat.activeThreadId;
    if (!threadId) return true;

    setBusy(true);
    setError(null);
    try {
      const session = await abandonTutorSession(threadId);
      updateChat(chat.id, (current) => reconcileSession(current, session));
      return true;
    } catch (caught) {
      setActiveChatId(chat.id);
      saveActiveChatId(chat.id);
      setError(errorMessage(caught));
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function createChat() {
    if (busy) return;
    const activeLesson = [...chats]
      .sort(
        (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
      )
      .find((chat) => Boolean(chat.activeThreadId));

    if (activeLesson) {
      const confirmed = window.confirm(
        "End the active lesson and start a new chat? The unfinished attempt will be retained as aborted.",
      );
      if (!confirmed || !(await endActiveLesson(activeLesson))) return;
    }

    const chat = createEmptyChat();
    setChats((current) => [chat, ...current]);
    setActiveChatId(chat.id);
    saveActiveChatId(chat.id);
    setSidebarOpen(false);
    setError(null);
  }

  function selectChat(chatId: string) {
    setActiveChatId(chatId);
    saveActiveChatId(chatId);
    setSidebarOpen(false);
    setError(null);
  }

  async function deleteChat(chatId: string) {
    if (busy) return;
    const chat = chats.find((item) => item.id === chatId);
    if (!chat) return;
    if (chat.activeThreadId) {
      const confirmed = window.confirm(
        `End the active lesson and delete "${chat.title}"? The unfinished attempt record will be retained as aborted.`,
      );
      if (!confirmed || !(await endActiveLesson(chat))) return;
    } else if (!window.confirm(`Delete "${chat.title}"?`)) {
      return;
    }

    const remaining = chats.filter((item) => item.id !== chatId);
    if (remaining.length === 0) {
      const replacement = createEmptyChat();
      setChats([replacement]);
      setActiveChatId(replacement.id);
      saveActiveChatId(replacement.id);
      return;
    }

    setChats(remaining);
    if (activeChatId === chatId) {
      const fallback = [...remaining].sort(
        (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
      )[0];
      setActiveChatId(fallback.id);
      saveActiveChatId(fallback.id);
    }
  }

  function updateChat(
    chatId: string,
    updater: (chat: ChatConversation) => ChatConversation,
  ) {
    setChats((current) =>
      current.map((chat) => (chat.id === chatId ? updater(chat) : chat)),
    );
  }

  function attachActiveSession(session: TutorSessionResponse) {
    const existing = chats.find(
      (chat) => chat.activeThreadId === session.thread_id || chat.session?.thread_id === session.thread_id,
    );
    const destination = existing ?? {
      ...createEmptyChat(),
      title: "Recovered lesson",
    };
    const recovered = reconcileSession(destination, session);

    setChats((current) =>
      existing
        ? current.map((chat) => (chat.id === existing.id ? recovered : chat))
        : [recovered, ...current],
    );
    setActiveChatId(destination.id);
    saveActiveChatId(destination.id);
    setError(null);
  }

  async function sendQuestion(question: string) {
    if (!learner || !activeChat || startRequestInFlightRef.current) return;

    const otherActiveLesson = [...chats]
      .sort(
        (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
      )
      .find((chat) => chat.id !== activeChat.id && Boolean(chat.activeThreadId));

    if (otherActiveLesson) {
      setActiveChatId(otherActiveLesson.id);
      saveActiveChatId(otherActiveLesson.id);
      setError("Continue and finish this active lesson before starting another one.");
      return;
    }

    if (activeChat.activeThreadId) {
      setError("Finish the current lesson before asking a new question.");
      return;
    }

    const chatId = activeChat.id;
    startRequestInFlightRef.current = true;
    setBusy(true);
    setError(null);

    try {
      try {
        const activeSession = await getActiveTutorSession();
        if (activeSession) {
          attachActiveSession(activeSession);
          return;
        }
      } catch (caught) {
        if (!(caught instanceof ApiError) || caught.status !== 404) throw caught;
      }

      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        type: "user",
        createdAt: nowIso(),
        content: question,
      };

      updateChat(chatId, (chat) => ({
        ...chat,
        title: chat.messages.length === 0 ? titleFromQuestion(question) : chat.title,
        messages: [...chat.messages, userMessage],
        updatedAt: nowIso(),
      }));

      const session = await startTutorSession({
        age: learner.age,
        question,
        topic: learner.topic,
        subtopic: learner.subtopic.trim() || null,
        ...(learner.targetSkill.trim()
          ? { target_skill: learner.targetSkill.trim() }
          : {}),
        relevant_history: [],
        previous_errors: [],
        previous_strategies: [],
      });

      updateChat(chatId, (chat) => reconcileSession(chat, session));
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 409) {
        try {
          const activeSession = await getActiveTutorSession();
          if (activeSession) {
            attachActiveSession(activeSession);
            return;
          }
        } catch (recoveryError) {
          if (!(recoveryError instanceof ApiError) || recoveryError.status !== 404) {
            setError(errorMessage(recoveryError));
            return;
          }
        }
      }
      setError(errorMessage(caught));
    } finally {
      startRequestInFlightRef.current = false;
      setBusy(false);
    }
  }

  async function submitStudentTurn(responseText: string) {
    if (
      !activeChat?.session ||
      !activeChat.activeThreadId ||
      activeChat.session.status !== "student_response_required"
    ) {
      return;
    }

    const threadId = activeChat.activeThreadId;
    const chatId = activeChat.id;
    setBusy(true);
    setError(null);

    const studentMessage: ChatMessage = {
      id: crypto.randomUUID(),
      type: "user",
      createdAt: nowIso(),
      content: responseText,
      threadId,
    };

    updateChat(chatId, (chat) => ({
      ...chat,
      messages: [...chat.messages, studentMessage],
      updatedAt: nowIso(),
    }));

    try {
      const session = await submitTutorTurn(threadId, { response: responseText });
      updateChat(chatId, (chat) => reconcileSession(chat, session));
    } catch (caught) {
      try {
        const latest = await getTutorSession(threadId);
        updateChat(chatId, (chat) => reconcileSession(chat, latest));
        setError(errorMessage(caught));
      } catch {
        setError(errorMessage(caught));
      }
    } finally {
      setBusy(false);
    }
  }

  async function submitAssessment(messageId: string, payload: TutorAnswerPayload) {
    if (!activeChat?.session || !activeChat.activeThreadId) return;

    const threadId = activeChat.activeThreadId;
    const chatId = activeChat.id;
    setBusy(true);
    setError(null);

    updateChat(chatId, (chat) => ({
      ...chat,
      messages: chat.messages.map((message) =>
        message.id === messageId && message.type === "assessment"
          ? { ...message, submitted: true, answers: payload.answers }
          : message,
      ),
      updatedAt: nowIso(),
    }));

    try {
      const session = await submitTutorAnswers(threadId, payload);
      updateChat(chatId, (chat) => reconcileSession(chat, session));
    } catch (caught) {
      try {
        const latest = await getTutorSession(threadId);
        updateChat(chatId, (chat) => reconcileSession(chat, latest));
        if (latest.status === "recovery_required") {
          setError("Your answers were saved. Resume the lesson to continue from the saved checkpoint.");
        } else {
          setError(errorMessage(caught));
        }
      } catch {
        setError(errorMessage(caught));
      }
    } finally {
      setBusy(false);
    }
  }

  async function resumeActiveLesson() {
    if (!activeChat?.session) return;
    const threadId = activeChat.session.thread_id;
    const chatId = activeChat.id;
    setBusy(true);
    setError(null);

    try {
      const session = await resumeTutorSession(threadId);
      updateChat(chatId, (chat) => {
        const withoutRecovery = chat.messages.filter(
          (message) => !(message.type === "recovery" && message.threadId === threadId),
        );
        return reconcileSession({ ...chat, messages: withoutRecovery }, session);
      });
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setBusy(false);
    }
  }

  function saveProfile(next: LearnerSetup) {
    saveLearnerSetup(next);
    setLearner(next);
    setSettingsOpen(false);
  }

  function navigate(next: AppPage) {
    const path = next === "profile" ? "/profile" : "/";
    if (window.location.pathname !== path) window.history.pushState({}, "", path);
    setPage(next);
    setSidebarOpen(false);
    setSettingsOpen(false);
  }

  async function practiceRecommendedSkill(skill: string): Promise<void> {
    if (!learner) throw new Error("Your learner context is not ready.");
    if (startRequestInFlightRef.current) {
      throw new Error("Another lesson is already being prepared.");
    }

    startRequestInFlightRef.current = true;
    setBusy(true);
    setError(null);

    try {
      const activeSession = await getActiveTutorSession();
      if (activeSession) {
        throw new ApiError(
          409,
          "Finish or abandon your active lesson before starting this recommendation.",
        );
      }

      const result = await startRecommendedPractice({ target_skill: skill });
      const nextLearner = { ...learner, targetSkill: result.target_skill };
      saveLearnerSetup(nextLearner);
      setLearner(nextLearner);

      const userMessage: ChatMessage = {
        id: crypto.randomUUID(),
        type: "user",
        createdAt: nowIso(),
        content: result.problem,
      };
      const destination: ChatConversation = {
        ...createEmptyChat(),
        title: titleFromQuestion(result.problem),
        messages: [userMessage],
      };
      const started = reconcileSession(destination, result.session);

      setChats((current) => [started, ...current]);
      setActiveChatId(started.id);
      saveActiveChatId(started.id);
      navigate("tutor");
    } finally {
      startRequestInFlightRef.current = false;
      setBusy(false);
    }
  }

  async function logout() {
    const confirmed = window.confirm(
      "Log out and clear this learner's chats from this browser? Saved learning records will not be deleted.",
    );
    if (!confirmed) return;

    let logoutFailed = false;
    try {
      await logoutUser();
    } catch {
      logoutFailed = true;
    } finally {
      clearStudentHint();
      clearLearnerSetup();
      saveChats([]);
      saveActiveChatId(null);
      setAuthUser(null);
      setAuthStatus("unauthenticated");
      setAuthNotice(
        logoutFailed
          ? "Local sign-out completed, but the authentication service could not confirm logout."
          : "You have been signed out.",
      );
      setLearner(null);
      setChats([]);
      setActiveChatId(null);
      setSidebarOpen(false);
      setSettingsOpen(false);
      setError(null);
      window.history.replaceState({}, "", "/");
      setPage("tutor");
    }
  }

  if (authStatus === "checking") {
    return (
      <main className="flex min-h-screen items-center justify-center bg-zinc-50 text-sm text-zinc-500">
        Restoring your AdaptMath session…
      </main>
    );
  }

  if (authStatus === "unauthenticated" || !authUser) {
    return (
      <AuthScreen
        notice={authNotice}
        onAuthenticated={activateAuthenticatedUser}
      />
    );
  }

  if (!learner) {
    return (
      <main className="flex min-h-screen items-center justify-center bg-zinc-50 text-sm text-zinc-500">
        Opening your Tutor workspace…
      </main>
    );
  }

  if (page === "profile") {
    return (
      <>
        <StudentProfilePage
          onBack={() => navigate("tutor")}
          onOpenSettings={() => setSettingsOpen(true)}
          onStartPracticing={practiceRecommendedSkill}
        />
        <ProfileDialog
          open={settingsOpen}
          identity={authUser}
          learner={learner}
          onClose={() => setSettingsOpen(false)}
          onSave={saveProfile}
        />
      </>
    );
  }

  const sessionStatus = activeChat?.session?.status ?? null;
  const waitingForExternalMove = sessionStatus === "pedagogical_move_required";
  const waitingForStudent = sessionStatus === "student_response_required";
  const waitingForAssessment = sessionStatus === "assessment_required";
  const waitingForRecovery = sessionStatus === "recovery_required";
  const hasActiveLesson = Boolean(activeChat?.activeThreadId);

  const composerDisabled =
    backendStatus === "offline" ||
    waitingForExternalMove ||
    waitingForAssessment ||
    waitingForRecovery ||
    (hasActiveLesson && !waitingForStudent);

  const composerPlaceholder = waitingForStudent
    ? "Reply to your tutor"
    : waitingForExternalMove
      ? "Preparing the next teaching step…"
      : waitingForAssessment
        ? "Complete the 3-question understanding check above"
        : waitingForRecovery
          ? "Resume the saved lesson above"
          : "Message AdaptMath";

  const messages = activeChat?.messages ?? [];

  return (
    <div className="flex h-screen overflow-hidden bg-white text-zinc-900">
      <ChatSidebar
        chats={chats}
        activeChatId={activeChatId}
        learner={learner}
        identity={authUser}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={createChat}
        onSelectChat={selectChat}
        onDeleteChat={deleteChat}
        onOpenProfile={() => navigate("profile")}
        onLogout={logout}
      />

      <section className="flex min-w-0 flex-1 flex-col bg-white">
        <header className="flex h-14 shrink-0 items-center justify-between border-b border-zinc-100 px-3 sm:px-4">
          <div className="flex min-w-0 items-center gap-2">
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              className="rounded-lg p-2 text-zinc-600 hover:bg-zinc-100 lg:hidden"
              aria-label="Open chat history"
            >
              <Menu className="h-5 w-5" aria-hidden="true" />
            </button>
            <button
              type="button"
              onClick={createChat}
              className="hidden rounded-lg p-2 text-zinc-600 hover:bg-zinc-100 sm:inline-flex lg:hidden"
              aria-label="New chat"
            >
              <Plus className="h-5 w-5" aria-hidden="true" />
            </button>
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold text-zinc-900">
                {activeChat?.title || "New chat"}
              </div>
              <div className="truncate text-[11px] text-zinc-400">
                {learner.targetSkill
                  ? `Focus: ${learner.targetSkill}`
                  : `${learner.topic}${learner.subtopic ? ` · ${learner.subtopic}` : ""}`}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div
              className={`hidden items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-medium sm:flex ${
                backendStatus === "ready"
                  ? "bg-emerald-50 text-emerald-700"
                  : backendStatus === "checking"
                    ? "bg-zinc-100 text-zinc-500"
                    : "bg-rose-50 text-rose-700"
              }`}
            >
              <span
                className={`h-1.5 w-1.5 rounded-full ${
                  backendStatus === "ready"
                    ? "bg-emerald-500"
                    : backendStatus === "checking"
                      ? "bg-zinc-400"
                      : "bg-rose-500"
                }`}
              />
              {backendStatus === "ready" ? "Ready" : backendStatus === "checking" ? "Checking" : "Offline"}
            </div>
            <button
              type="button"
              onClick={() => setSettingsOpen(true)}
              className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-100"
              aria-label="Learner settings"
            >
              <Settings2 className="h-4.5 w-4.5" aria-hidden="true" />
            </button>
          </div>
        </header>

        <div ref={scrollerRef} className="min-h-0 flex-1 overflow-y-auto scroll-smooth">
          {messages.length === 0 ? (
            <EmptyChat onExample={sendQuestion} />
          ) : (
            <ChatMessages
              messages={messages}
              busy={busy}
              onSubmitAssessment={submitAssessment}
              onResume={resumeActiveLesson}
            />
          )}

          {(busy || waitingForExternalMove) && messages.length > 0 && (
            <div className="mx-auto w-full max-w-3xl px-4 pb-8">
              <div className="chat-row assistant-row py-2">
                <div className="assistant-avatar" aria-hidden="true">A</div>
                <div className="pt-1.5">
                  <div className="flex items-center gap-1.5 text-zinc-400" aria-label="AdaptMath is preparing the next teaching step">
                    <span className="thinking-dot" />
                    <span className="thinking-dot animation-delay-150" />
                    <span className="thinking-dot animation-delay-300" />
                  </div>
                  {waitingForExternalMove && !busy && (
                    <div className="mt-1 text-xs text-zinc-400">Preparing the next teaching step…</div>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="shrink-0 bg-gradient-to-t from-white via-white to-white/70 px-3 pb-3 pt-2 sm:px-5 sm:pb-4">
          {error && (
            <div className="mx-auto mb-2 max-w-3xl rounded-xl bg-rose-50 px-3.5 py-2.5 text-sm leading-5 text-rose-700" role="alert">
              {error}
            </div>
          )}
          <ChatComposer
            busy={busy}
            disabled={composerDisabled}
            placeholder={composerPlaceholder}
            onSubmit={waitingForStudent ? submitStudentTurn : sendQuestion}
          />
        </div>
      </section>

      <ProfileDialog
        open={settingsOpen}
        identity={authUser}
        learner={learner}
        onClose={() => setSettingsOpen(false)}
        onSave={saveProfile}
      />
    </div>
  );
}
