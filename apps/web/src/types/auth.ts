export interface AuthenticatedUser {
  user_id: string;
  username: string;
  role: "STUDENT";
  student_id: string;
  age?: number | null;
}

export interface SignupPayload {
  username: string;
  date_of_birth: string;
  password: string;
  confirm_password: string;
}

export interface SignupResponse {
  username: string;
  student_id: string;
  age: number;
  role: "STUDENT";
}

export interface LoginPayload {
  username: string;
  password: string;
}

interface LoginResponse {
  token: string;
  username: string;
  role: "STUDENT";
  student_id: string;
  age?: number | null;
}

export interface AuthenticatedSession {
  token: string;
  user: AuthenticatedUser;
}

export type { LoginResponse };
