import { useState, type FormEvent } from "react";
import { ArrowRight, BookOpen, LockKeyhole, Sparkles, UserPlus } from "lucide-react";

import { AuthApiError, login, signup } from "../api/auth";
import type { AuthenticatedUser } from "../types/auth";
import { ThemeToggle } from "./ThemeToggle";

interface AuthScreenProps {
  notice?: string | null;
  onAuthenticated: (user: AuthenticatedUser) => void;
}

export function AuthScreen({ notice, onAuthenticated }: AuthScreenProps) {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [username, setUsername] = useState("");
  const [dateOfBirth, setDateOfBirth] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedUsername = username.trim();
    if (!normalizedUsername || !password) return;
    if (mode === "signup" && password.length < 8) {
      setError("Password must be at least 8 characters long.");
      return;
    }
    if (mode === "signup" && password !== confirmPassword) {
      setError("Password and confirmation password do not match.");
      return;
    }

    setBusy(true);
    setError(null);
    try {
      if (mode === "signup") {
        await signup({
          username: normalizedUsername,
          date_of_birth: dateOfBirth,
          password,
          confirm_password: confirmPassword,
        });
      }
      const session = await login({ username: normalizedUsername, password });
      onAuthenticated(session.user);
    } catch (caught) {
      setError(
        caught instanceof AuthApiError
          ? caught.message
          : "Authentication service is unavailable. Please try again.",
      );
    } finally {
      setBusy(false);
    }
  }

  function switchMode(next: "login" | "signup") {
    if (next === mode) return;
    setMode(next);
    setUsername("");
    setDateOfBirth("");
    setPassword("");
    setConfirmPassword("");
    setError(null);
  }

  return (
    <main className="relative h-[100dvh] overflow-x-hidden overflow-y-auto bg-gradient-to-br from-indigo-50 via-white to-cyan-50 transition-colors dark:from-slate-950 dark:via-slate-950 dark:to-indigo-950/60">
      <div className="fixed right-4 top-4 z-20 sm:right-6 sm:top-6">
        <ThemeToggle />
      </div>
      <div className="pointer-events-none absolute -left-24 top-1/4 h-72 w-72 rounded-full bg-violet-200/40 blur-3xl dark:bg-violet-700/10" />
      <div className="pointer-events-none absolute -right-24 bottom-0 h-80 w-80 rounded-full bg-cyan-200/50 blur-3xl dark:bg-cyan-600/10" />

      <div className="relative flex min-h-full items-center justify-center px-4 py-20">
        <div className="w-full max-w-md">
        <div className="mb-7 text-center">
          <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-indigo-600 to-violet-600 text-white shadow-lg shadow-indigo-500/20">
            {mode === "login" ? <LockKeyhole className="h-7 w-7" /> : <UserPlus className="h-7 w-7" />}
          </div>
          <div className="mt-5 flex items-center justify-center gap-2 text-sm font-bold uppercase tracking-[0.18em] text-indigo-600 dark:text-indigo-300">
            <Sparkles className="h-4 w-4" /> AdaptMath
          </div>
          <h1 className="mt-2 text-3xl font-bold tracking-tight text-slate-950 dark:text-white">
            {mode === "login" ? "Ready to learn?" : "Start your maths journey"}
          </h1>
          <p className="mt-2 text-sm leading-6 text-slate-500 dark:text-slate-400">
            {mode === "login" ? "Sign in and pick up right where you left off." : "Create an account for lessons that adapt to you."}
          </p>
        </div>

        <div className="mb-4 grid grid-cols-2 rounded-2xl bg-indigo-100/70 p-1.5 text-sm dark:bg-slate-800">
          <button type="button" onClick={() => switchMode("login")} className={`rounded-xl px-3 py-2.5 font-semibold transition ${mode === "login" ? "bg-white text-indigo-700 shadow-sm dark:bg-slate-700 dark:text-indigo-200" : "text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200"}`}>Sign in</button>
          <button type="button" onClick={() => switchMode("signup")} className={`rounded-xl px-3 py-2.5 font-semibold transition ${mode === "signup" ? "bg-white text-indigo-700 shadow-sm dark:bg-slate-700 dark:text-indigo-200" : "text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200"}`}>Create account</button>
        </div>

        <form key={mode} onSubmit={submit} autoComplete={mode === "signup" ? "off" : "on"} className="grid gap-4 rounded-3xl border border-white/80 bg-white/90 p-6 shadow-xl shadow-indigo-900/5 backdrop-blur dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-black/20">
          {(notice || error) && (
            <div className={`rounded-xl px-3.5 py-2.5 text-sm ${error ? "bg-rose-50 text-rose-700 dark:bg-rose-500/10 dark:text-rose-300" : "bg-amber-50 text-amber-800 dark:bg-amber-500/10 dark:text-amber-200"}`} role="alert">
              {error || notice}
            </div>
          )}
          <label className="grid gap-1.5 text-sm font-semibold text-slate-700 dark:text-slate-200">
            Username
            <input className="chat-field" name={mode === "login" ? "login-username" : "new-account-username"} autoComplete={mode === "login" ? "section-login username" : "off"} autoCapitalize="none" spellCheck={false} value={username} maxLength={255} onChange={(event) => setUsername(event.target.value)} required />
          </label>
          {mode === "signup" && (
            <label className="grid gap-1.5 text-sm font-semibold text-slate-700 dark:text-slate-200">
              Date of birth
              <input className="chat-field" name="new-account-date-of-birth" type="date" autoComplete="off" value={dateOfBirth} max={new Date().toISOString().slice(0, 10)} onChange={(event) => setDateOfBirth(event.target.value)} required />
            </label>
          )}
          <label className="grid gap-1.5 text-sm font-semibold text-slate-700 dark:text-slate-200">
            Password
            <input className="chat-field" name={mode === "login" ? "login-password" : "new-account-password"} type="password" autoComplete={mode === "login" ? "section-login current-password" : "new-password"} value={password} minLength={mode === "signup" ? 8 : 1} maxLength={255} onChange={(event) => setPassword(event.target.value)} required />
          </label>
          {mode === "signup" && (
            <label className="grid gap-1.5 text-sm font-semibold text-slate-700 dark:text-slate-200">
              Confirm password
              <input className="chat-field" name="new-account-password-confirmation" type="password" autoComplete="new-password" value={confirmPassword} minLength={8} maxLength={255} onChange={(event) => setConfirmPassword(event.target.value)} required />
            </label>
          )}
          <button type="submit" disabled={busy} className="mt-1 flex items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-indigo-600 to-violet-600 px-4 py-3.5 text-sm font-bold text-white shadow-lg shadow-indigo-500/20 transition hover:-translate-y-0.5 hover:shadow-xl disabled:translate-y-0 disabled:opacity-40">
            {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
            {!busy && <ArrowRight className="h-4 w-4" aria-hidden="true" />}
          </button>
        </form>
        <p className="mt-5 flex items-center justify-center gap-2 text-center text-xs text-slate-400 dark:text-slate-500">
          <BookOpen className="h-3.5 w-3.5" /> Learn at your pace, one step at a time.
        </p>
        </div>
      </div>
    </main>
  );
}
