import type {
  AuthenticatedSession,
  AuthenticatedUser,
  LoginPayload,
  LoginResponse,
  SignupPayload,
  SignupResponse,
} from "../types/auth";

const MEMORY_API_BASE_URL = (
  import.meta.env.VITE_MEMORY_API_BASE_URL || "http://127.0.0.1:8400"
).replace(/\/$/, "");

const TOKEN_KEY = "adaptmath.authToken.v1";
const IDENTITY_HINT_KEY = "adaptmath.authStudentHint.v1";
export const AUTH_INVALID_EVENT = "adaptmath:auth-invalid";

export class AuthApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "AuthApiError";
    this.status = status;
  }
}

async function authRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${MEMORY_API_BASE_URL}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let message = `Authentication request failed with status ${response.status}.`;
    try {
      const payload = (await response.json()) as {
        detail?: unknown;
        message?: unknown;
      };
      if (typeof payload.detail === "string") message = payload.detail;
      else if (typeof payload.message === "string") message = payload.message;
    } catch {
      // Keep the generic message when the service did not return JSON.
    }
    throw new AuthApiError(response.status, message);
  }

  return (await response.json()) as T;
}

export function getStoredAuthToken(): string | null {
  return window.localStorage.getItem(TOKEN_KEY);
}

export function clearStoredAuthToken(): void {
  window.localStorage.removeItem(TOKEN_KEY);
}

export function getStoredStudentHint(): string | null {
  return window.localStorage.getItem(IDENTITY_HINT_KEY);
}

export function saveStudentHint(studentId: string): void {
  window.localStorage.setItem(IDENTITY_HINT_KEY, studentId);
}

export function clearStudentHint(): void {
  window.localStorage.removeItem(IDENTITY_HINT_KEY);
}

export function notifyInvalidAuthentication(): void {
  clearStoredAuthToken();
  window.dispatchEvent(new Event(AUTH_INVALID_EVENT));
}

export function signup(payload: SignupPayload): Promise<SignupResponse> {
  return authRequest<SignupResponse>("/auth/signup", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function me(token = getStoredAuthToken()): Promise<AuthenticatedUser> {
  if (!token) throw new AuthApiError(401, "Authentication required.");
  return authRequest<AuthenticatedUser>("/auth/me", {
    headers: { Authorization: `Bearer ${token}` },
  });
}

export async function login(payload: LoginPayload): Promise<AuthenticatedSession> {
  const response = await authRequest<LoginResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  window.localStorage.setItem(TOKEN_KEY, response.token);
  try {
    const user = await me(response.token);
    return { token: response.token, user };
  } catch (error) {
    clearStoredAuthToken();
    throw error;
  }
}

export async function restoreAuthentication(): Promise<AuthenticatedUser | null> {
  const token = getStoredAuthToken();
  if (!token) return null;
  try {
    return await me(token);
  } catch {
    clearStoredAuthToken();
    return null;
  }
}

export async function logout(): Promise<void> {
  const token = getStoredAuthToken();
  try {
    if (token) {
      await authRequest<{ message: string }>("/auth/logout", {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });
    }
  } finally {
    clearStoredAuthToken();
  }
}
