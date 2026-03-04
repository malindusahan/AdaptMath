import { useState, type FormEvent } from "react";
import { ArrowRight, LockKeyhole, UserPlus } from "lucide-react";

import { AuthApiError, login, signup } from "../api/auth";
import type { AuthenticatedUser } from "../types/auth";

interface AuthScreenProps {
  notice?: string | null;
  onAuthenticated: (user: AuthenticatedUser) => void;
}

export function AuthScreen({ notice, onAuthenticated }: AuthScreenProps) {
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [username, setUsername] = useState("");
  const [dateOfBirth, setDateOfBirth] = useState("2011-06-15");
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
    setMode(next);
    setError(null);
    setConfirmPassword("");
  }

  return (
    <main className="flex min-h-screen items-center justify-center bg-zinc-50 px-4 py-10">
      <div className="w-full max-w-md">
        <div className="mb-7 text-center">
          <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-zinc-900 text-white">
            {mode === "login" ? <LockKeyhole className="h-5 w-5" /> : <UserPlus className="h-5 w-5" />}
          </div>
          <h1 className="mt-5 text-2xl font-semibold tracking-tight text-zinc-950">AdaptMath</h1>
          <p className="mt-2 text-sm text-zinc-500">
            {mode === "login" ? "Sign in to continue learning." : "Create your learner account."}
          </p>
        </div>

        <div className="mb-4 grid grid-cols-2 rounded-xl bg-zinc-200/70 p-1 text-sm">
          <button type="button" onClick={() => switchMode("login")} className={`rounded-lg px-3 py-2 font-medium ${mode === "login" ? "bg-white text-zinc-950 shadow-sm" : "text-zinc-500"}`}>Login</button>
          <button type="button" onClick={() => switchMode("signup")} className={`rounded-lg px-3 py-2 font-medium ${mode === "signup" ? "bg-white text-zinc-950 shadow-sm" : "text-zinc-500"}`}>Register</button>
        </div>

        <form onSubmit={submit} className="grid gap-4 rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm">
          {(notice || error) && (
            <div className={`rounded-xl px-3.5 py-2.5 text-sm ${error ? "bg-rose-50 text-rose-700" : "bg-amber-50 text-amber-800"}`} role="alert">
              {error || notice}
            </div>
          )}
          <label className="grid gap-1.5 text-sm font-medium text-zinc-700">
            Username
            <input className="chat-field" autoComplete="username" value={username} maxLength={255} onChange={(event) => setUsername(event.target.value)} required />
          </label>
          {mode === "signup" && (
            <label className="grid gap-1.5 text-sm font-medium text-zinc-700">
              Date of birth
              <input className="chat-field" type="date" value={dateOfBirth} max={new Date().toISOString().slice(0, 10)} onChange={(event) => setDateOfBirth(event.target.value)} required />
            </label>
          )}
          <label className="grid gap-1.5 text-sm font-medium text-zinc-700">
            Password
            <input className="chat-field" type="password" autoComplete={mode === "login" ? "current-password" : "new-password"} value={password} minLength={mode === "signup" ? 8 : 1} maxLength={255} onChange={(event) => setPassword(event.target.value)} required />
          </label>
          {mode === "signup" && (
            <label className="grid gap-1.5 text-sm font-medium text-zinc-700">
              Confirm password
              <input className="chat-field" type="password" autoComplete="new-password" value={confirmPassword} minLength={8} maxLength={255} onChange={(event) => setConfirmPassword(event.target.value)} required />
            </label>
          )}
          <button type="submit" disabled={busy} className="mt-1 flex items-center justify-center gap-2 rounded-xl bg-zinc-900 px-4 py-3 text-sm font-medium text-white hover:bg-zinc-800 disabled:opacity-40">
            {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
            {!busy && <ArrowRight className="h-4 w-4" aria-hidden="true" />}
          </button>
        </form>
      </div>
    </main>
  );
}
