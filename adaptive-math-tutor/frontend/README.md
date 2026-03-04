# AdaptMath Frontend — Chat Workspace

This is the single learner-facing React + TypeScript interface for the local AdaptMath stack.

## Current UX

- Login and registration use the Memory service's existing `/auth/*` protocol.
- Registration is followed by login and opens the Tutor workspace directly; a valid account session gates the Tutor UI.
- Tutor requests carry the Memory-issued Bearer token. The Tutor backend validates it through Memory `/auth/me` and derives the canonical student ID server-side.
- Startup revalidates a stored token through `/auth/me`; invalid or restart-lost sessions return the learner to Login.
- Logout invalidates the Memory session before clearing local auth and Tutor state.
- ChatGPT-style two-column chat workspace.
- Left sidebar with **New chat** and recent chat history.
- Multiple learner questions can live inside one frontend chat.
- Each learner question currently starts a separate backend tutoring thread because the standalone backend is question/session-oriented.
- The frontend groups those backend tutoring sessions into one continuous learner conversation.
- Inline tutor explanations with Markdown + KaTeX mathematics.
- Inline two-question assessments, evaluation feedback, adaptive reteaching, recovery and completion states.
- Chat history persists in browser `localStorage` for the standalone build.
- Active backend tutoring sessions can still recover from SQLite checkpoints through the backend APIs.

## Learner identity

The authenticated account supplies the learner ID. It is not editable in the UI. On first login, account age and neutral lesson defaults are initialized automatically so there is no separate setup screen. Age, topic, subtopic, and target skill remain editable from the learner-context settings dialog. Developer mode may display the resolved ID read-only.

The active local launcher uses these canonical service URLs by default:

- Adaptive Tutor frontend: `http://127.0.0.1:5173`
- Adaptive Tutor API: `http://127.0.0.1:8402`
- Memory/auth API: `http://127.0.0.1:8400`

Override the APIs with `VITE_TUTOR_API_BASE_URL` and `VITE_MEMORY_API_BASE_URL`. Do not put service API keys in browser variables.

## Prototype security limitations

- User sessions live in Memory-service process memory and are lost on restart.
- The browser token is stored in `localStorage`. This is acceptable only for this local research prototype; production should prefer secure HttpOnly cookies.
- There is no account recovery, email or identity verification, login throttling, multi-factor authentication, or production-grade password hardening.
- Browser chat and setup state is local and is cleared when a different authenticated student is detected; server-side thread ownership remains the security boundary.

## Run

```powershell
cd adaptive-math-tutor\frontend
npm install
Copy-Item .env.example .env -Force
npm run build
npm run dev
```

For the integrated stack, run the workspace launcher instead of starting the standalone Memory frontend. The standalone Memory UI remains a development reference and is not required by learners.
