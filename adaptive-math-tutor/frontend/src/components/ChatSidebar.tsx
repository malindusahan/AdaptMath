import {
  MessageSquareText,
  Plus,
  Settings2,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import type { ChatConversation } from "../types/chat";
import type { LearnerSetup } from "../types/tutor";

interface ChatSidebarProps {
  chats: ChatConversation[];
  activeChatId: string | null;
  learner: LearnerSetup;
  open: boolean;
  onClose: () => void;
  onNewChat: () => void;
  onSelectChat: (chatId: string) => void;
  onDeleteChat: (chatId: string) => void;
  onOpenSettings: () => void;
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
      <div className="px-3 pb-2 text-xs font-semibold text-zinc-500">{label}</div>
      <div className="grid gap-0.5">
        {chats.map((chat) => (
          <div
            key={chat.id}
            className={`group flex min-w-0 items-center rounded-lg px-2 transition ${
              activeChatId === chat.id ? "bg-zinc-200/75" : "hover:bg-zinc-200/55"
            }`}
          >
            <button
              type="button"
              onClick={() => onSelectChat(chat.id)}
              className="flex min-w-0 flex-1 items-center gap-2.5 px-1 py-2.5 text-left text-sm text-zinc-800"
            >
              <MessageSquareText className="h-4 w-4 shrink-0 text-zinc-500" aria-hidden="true" />
              <span className="truncate">{chat.title}</span>
            </button>
            <button
              type="button"
              onClick={() => onDeleteChat(chat.id)}
              className="rounded-md p-1.5 text-zinc-400 opacity-0 transition hover:bg-zinc-300 hover:text-zinc-700 group-hover:opacity-100 focus:opacity-100"
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
  open,
  onClose,
  onNewChat,
  onSelectChat,
  onDeleteChat,
  onOpenSettings,
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
          className="fixed inset-0 z-30 bg-black/25 lg:hidden"
          aria-label="Close sidebar"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[282px] flex-col border-r border-zinc-200 bg-[#f7f7f8] transition-transform duration-200 lg:static lg:translate-x-0 ${
          open ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <div className="flex items-center justify-between px-3 pb-2 pt-3">
          <div className="flex items-center gap-2 px-2 font-semibold text-zinc-900">
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-zinc-900 text-white">
              <Sparkles className="h-4 w-4" aria-hidden="true" />
            </span>
            AdaptMath
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-2 text-zinc-500 hover:bg-zinc-200 lg:hidden"
            aria-label="Close sidebar"
          >
            <X className="h-5 w-5" aria-hidden="true" />
          </button>
        </div>

        <div className="px-3 pb-3">
          <button
            type="button"
            onClick={onNewChat}
            className="flex w-full items-center gap-2.5 rounded-xl border border-zinc-300 bg-white px-3 py-2.5 text-sm font-medium text-zinc-900 shadow-sm transition hover:bg-zinc-50"
          >
            <Plus className="h-4 w-4" aria-hidden="true" />
            New chat
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-4 scrollbar-thin">
          {sorted.length === 0 ? (
            <div className="px-3 py-8 text-center text-sm leading-6 text-zinc-500">
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

        <div className="border-t border-zinc-200 p-2.5">
          <button
            type="button"
            onClick={onOpenSettings}
            className="flex w-full items-center gap-3 rounded-xl px-2.5 py-2.5 text-left transition hover:bg-zinc-200/70"
          >
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-zinc-900 text-sm font-semibold text-white">
              {learner.studentId.slice(0, 1).toUpperCase() || "S"}
            </div>
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm font-medium text-zinc-900">{learner.studentId}</div>
              <div className="truncate text-xs text-zinc-500">
                Age {learner.age} · {learner.topic}
              </div>
            </div>
            <Settings2 className="h-4 w-4 text-zinc-500" aria-hidden="true" />
          </button>
        </div>
      </aside>
    </>
  );
}
