import type {
  TutorAnswerPayload,
  TutorPracticeStartPayload,
  TutorPracticeStartResponse,
  TutorSessionResponse,
  TutorStartPayload,
  TutorStudentTurnPayload,
} from "../types/tutor";
import type { StudentProfile } from "../types/profile";
import {
  getStoredAuthToken,
  notifyInvalidAuthentication,
} from "./auth";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8402"
).replace(/\/$/, "");

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const token = getStoredAuthToken();
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    if (response.status === 401 && (path.startsWith("/tutor") || path === "/profile")) {
      notifyInvalidAuthentication();
    }
    let detail = `Request failed with status ${response.status}.`;

    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      } else if (payload.detail) {
        detail = JSON.stringify(payload.detail);
      }
    } catch {
      // Keep the safe generic message if the response body is not JSON.
    }

    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as T;
}

export function getStudentProfile(): Promise<StudentProfile> {
  return request<StudentProfile>("/profile");
}

export function startTutorSession(
  payload: TutorStartPayload,
): Promise<TutorSessionResponse> {
  return request<TutorSessionResponse>("/tutor/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function startRecommendedPractice(
  payload: TutorPracticeStartPayload,
): Promise<TutorPracticeStartResponse> {
  return request<TutorPracticeStartResponse>("/tutor/start-practice", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getTutorSession(
  threadId: string,
): Promise<TutorSessionResponse> {
  return request<TutorSessionResponse>(
    `/tutor/${encodeURIComponent(threadId)}`,
  );
}

export function getActiveTutorSession(): Promise<TutorSessionResponse | null> {
  return request<TutorSessionResponse | null>("/tutor/active");
}

export function abandonTutorSession(
  threadId: string,
): Promise<TutorSessionResponse> {
  return request<TutorSessionResponse>(
    `/tutor/${encodeURIComponent(threadId)}/abandon`,
    { method: "POST" },
  );
}

export function submitTutorTurn(
  threadId: string,
  payload: TutorStudentTurnPayload,
): Promise<TutorSessionResponse> {
  return request<TutorSessionResponse>(
    `/tutor/${encodeURIComponent(threadId)}/turn`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

export function submitTutorAnswers(
  threadId: string,
  payload: TutorAnswerPayload,
): Promise<TutorSessionResponse> {
  return request<TutorSessionResponse>(
    `/tutor/${encodeURIComponent(threadId)}/answers`,
    {
      method: "POST",
      body: JSON.stringify(payload),
    },
  );
}

export function resumeTutorSession(
  threadId: string,
): Promise<TutorSessionResponse> {
  return request<TutorSessionResponse>(
    `/tutor/${encodeURIComponent(threadId)}/resume`,
    { method: "POST" },
  );
}

export async function getHealth(): Promise<boolean> {
  try {
    await request<Record<string, unknown>>("/health");
    return true;
  } catch {
    return false;
  }
}

export async function getReady(): Promise<boolean> {
  try {
    await request<Record<string, unknown>>("/ready");
    return true;
  } catch {
    return false;
  }
}
