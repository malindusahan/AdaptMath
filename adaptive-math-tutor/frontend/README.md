# AdaptMath Frontend — Chat Workspace

This frontend is a learner-facing React + TypeScript interface for the AdaptMath backend.

## Current UX

- ChatGPT-style two-column chat workspace.
- Left sidebar with **New chat** and recent chat history.
- Multiple learner questions can live inside one frontend chat.
- Each learner question currently starts a separate backend tutoring thread because the standalone backend is question/session-oriented.
- The frontend groups those backend tutoring sessions into one continuous learner conversation.
- Inline tutor explanations with Markdown + KaTeX mathematics.
- Inline two-question assessments, evaluation feedback, adaptive reteaching, recovery and completion states.
- Chat history persists in browser `localStorage` for the standalone build.
- Active backend tutoring sessions can still recover from SQLite checkpoints through the backend APIs.

## Temporary integration fields

Learner ID, age, topic and subtopic are stored as the local learner context. They are intentionally isolated behind the learner settings dialog so that Profile/Auth and Topic APIs can replace them during team integration.

Long-term cross-session learning memory is not fabricated in this frontend. The backend request currently sends empty history/error/strategy arrays until the real Memory integration is connected.

## Run

```powershell
cd D:\adaptive-math-tutor\frontend
npm install
Copy-Item .env.example .env -Force
npm run build
npm run dev
```

Backend should be available at `http://127.0.0.1:8000` by default.
