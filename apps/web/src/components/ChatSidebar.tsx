import {
  LogOut,
  MessageSquareText,
  Plus,
  Sparkles,
  Trash2,
  UserRound,
  X,
} from "lucide-react";
import type { AuthenticatedUser } from "../types/auth";
import type { ChatConversation } from "../types/chat";
import type { LearnerSetup } from "../types/tutor";

interface ChatSidebarProps {
  chats: ChatConversation[];
  activeChatId: string | null;
  learner: LearnerSetup;
  identity: AuthenticatedUser;
  open: boolean;
  onClose: () => void;
  onNewChat: () => void;
  onSelectChat: (chatId: string) => void;
  onDeleteChat: (chatId: string) => void;
  onOpenProfile: () => void;
  onLogout: () => void;
}

function isToday(iso: string): boolean {
  const date = new Date(iso);
  const now = new Date();
  return (
    date.getFullYear() === now.getFullYear() &&
    date.getMonth() === now.getMonth() &&
    date.getDate() === now.getDate()
  );
}

function ChatGroup({
  label,
  chats,
  activeChatId,
  onSelectChat,
  onDeleteChat,
}: {
  label: string;
  chats: ChatConversation[];
  activeChatId: string | null;
  onSelectChat: (chatId: string) => void;
  onDeleteChat: (chatId: string) => void;
}) {
  if (chats.length === 0) return null;

  return (
    <div className="mb-5">
      <div className="px-3 pb-2 text-[11px] font-bold uppercase tracking-[0.14em] text-slate-400 dark:text-slate-500">{label}</div>
      <div className="grid gap-0.5">
        {chats.map((chat) => (
          <div
            key={chat.id}
            className={`group flex min-w-0 items-center rounded-lg px-2 transition ${
              activeChatId === chat.id
                ? "bg-indigo-100/80 dark:bg-indigo-500/15"
                : "hover:bg-indigo-50/80 dark:hover:bg-slate-800"
            }`}
          >
            <button
              type="button"
              onClick={() => onSelectChat(chat.id)}
              className="flex min-w-0 flex-1 items-center gap-2.5 px-1 py-2.5 text-left text-sm font-medium text-slate-700 dark:text-slate-200"
            >
              <MessageSquareText className="h-4 w-4 shrink-0 text-indigo-500 dark:text-indigo-300" aria-hidden="true" />
              <span className="truncate">{chat.title}</span>
              {chat.activeThreadId && (
                <span
                  className="h-2 w-2 shrink-0 rounded-full bg-emerald-500 ring-2 ring-emerald-500/15"
                  title="Lesson in progress"
                  aria-label="Lesson in progress"
                />
              )}
              {!chat.activeThreadId && chat.paused && (
                <span
                  className="h-2 w-2 shrink-0 rounded-full bg-amber-400 ring-2 ring-amber-400/15"
                  title="Lesson paused — open it to continue"
                  aria-label="Lesson paused"
                />
              )}
            </button>
            <button
              type="button"
              onClick={() => onDeleteChat(chat.id)}
              className="rounded-lg p-1.5 text-slate-400 opacity-0 transition hover:bg-white hover:text-rose-600 group-hover:opacity-100 focus:opacity-100 dark:hover:bg-slate-700 dark:hover:text-rose-300"
              aria-label={`Delete ${chat.title}`}
              title="Delete chat"
            >
              <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ChatSidebar({
  chats,
  activeChatId,
  learner,
  identity,
  open,
  onClose,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onOpenProfile,
  onLogout,
}: ChatSidebarProps) {
  const sorted = [...chats].sort(
    (a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime(),
  );
  const today = sorted.filter((chat) => isToday(chat.updatedAt));
  const earlier = sorted.filter((chat) => !isToday(chat.updatedAt));

  return (
    <>
      {open && (
        <button
          type="button"
          className="fixed inset-0 z-30 bg-slate-950/45 backdrop-blur-sm lg:hidden"
          aria-label="Close sidebar"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[292px] flex-col border-r border-indigo-100 bg-gradient-to-b from-indigo-50/90 via-[#f8faff] to-[#f8faff] transition-transform duration-200 dark:border-slate-800 dark:from-indigo-950/30 dark:via-slate-900 dark:to-slate-900 lg:static lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between px-3 pb-2 pt-3">
          <div className="flex items-center gap-2.5 px-2 font-bold text-slate-900 dark:text-white">
            <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-600 to-violet-600 text-white shadow-md shadow-indigo-500/20">
              <Sparkles className="h-4 w-4" aria-hidden="true" />
            </span>
            AdaptMath
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-xl p-2 text-slate-500 hover:bg-white/80 dark:text-slate-400 dark:hover:bg-slate-800 lg:hidden"
            aria-label="Close sidebar"
          >
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>

        <div className="px-3 pb-3">
          <button
            type="button"
            onClick={onNewChat}
            className="flex w-full items-center justify-center gap-2.5 rounded-2xl bg-gradient-to-r from-indigo-600 to-violet-600 px-3 py-3 text-sm font-bold text-white shadow-lg shadow-indigo-500/15 transition hover:-translate-y-0.5 hover:shadow-xl"
          >
            <Plus className="h-4 w-4" aria-hidden="true" />
            New chat
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-4 scrollbar-thin">
          {sorted.length === 0 ? (
            <div className="px-3 py-8 text-center text-sm leading-6 text-slate-500 dark:text-slate-400">
              Your recent maths chats will appear here.
            </div>
          ) : (
            <>
              <ChatGroup
                label="Today"
                chats={today}
                activeChatId={activeChatId}
                onSelectChat={onSelectChat}
                onDeleteChat={onDeleteChat}
              />
              <ChatGroup
                label="Earlier"
                chats={earlier}
                activeChatId={activeChatId}
                onSelectChat={onSelectChat}
                onDeleteChat={onDeleteChat}
              />
            </>
          )}
        </div>

        <div className="border-t border-indigo-100 p-2.5 dark:border-slate-800">
          <button
            type="button"
            onClick={onOpenProfile}
            aria-label="Open student profile"
            title="Student profile"
            className="flex w-full items-center gap-3 rounded-2xl px-2.5 py-2.5 text-left transition hover:bg-white/80 dark:hover:bg-slate-800"
          >
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-cyan-500 to-indigo-600 text-sm font-bold text-white shadow-sm">
              {identity.username.slice(0, 1).toUpperCase() || "S"}
            </div>
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-bold text-slate-900 dark:text-slate-100">{identity.username}</div>
              <div className="truncate text-xs text-slate-500 dark:text-slate-400">
                Age {learner.age} · {learner.topic}
              </div>
            </div>
            <UserRound className="h-4 w-4 text-slate-400" aria-hidden="true" />
          </button>
          <button
            type="button"
            onClick={onLogout}
            className="mt-1 flex w-full items-center gap-3 rounded-xl px-2.5 py-2.5 text-left text-sm font-medium text-slate-500 transition hover:bg-rose-50 hover:text-rose-700 dark:text-slate-400 dark:hover:bg-rose-500/10 dark:hover:text-rose-300"
          >
            <LogOut className="h-4 w-4" aria-hidden="true" />
            Log out
          </button>
        </div>
      </aside>
    </>
  );
}
