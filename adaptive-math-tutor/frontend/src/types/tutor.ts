export type SessionStatus =
  | "student_response_required"
  | "pedagogical_move_required"
  | "assessment_required"
  | "recovery_required"
  | "complete";

export interface AssessmentQuestion {
  question_id: string;
  question: string;
}

export interface EvaluationAnswer {
  question_id?: string;
  question?: string;
  student_answer?: string;
  is_correct?: boolean;
  feedback?: string;
  identified_error?: string;
}

export interface EvaluationResult {
  correct_answers: EvaluationAnswer[];
  wrong_answers: EvaluationAnswer[];
  identified_errors: string[];
  needs_reteaching: boolean;
  overall_feedback?: string | null;
}

export interface TutorSessionResponse {
  thread_id: string;
  status: SessionStatus;
  tutor_response?: string | null;
  route?: string | null;
  route_reason?: string | null;
  complexity_score?: number | null;
  student_response_message?: string | null;
  assessment_message?: string | null;
  pedagogical_move_message?: string | null;
  allowed_pedagogical_moves: string[];
  questions: AssessmentQuestion[];
  evaluation?: EvaluationResult | null;
  reteach_round: number;
  turn_count: number;
}

export interface LearnerHistoryItem {
  topic: string;
  event_type: string;
  details: Record<string, unknown>;
}

export interface TutorStartPayload {
  student_id: string;
  age: number;
  question: string;
  topic: string;
  subtopic?: string | null;
  target_skill: string;
  relevant_history: LearnerHistoryItem[];
  previous_errors: string[];
  previous_strategies: string[];
}

export interface TutorStudentTurnPayload {
  response: string;
}

export interface StudentAnswerPayload {
  question_id: string;
  answer: string;
}

export interface TutorAnswerPayload {
  answers: StudentAnswerPayload[];
}

export interface LearnerSetup {
  studentId: string;
  age: number;
  topic: string;
  subtopic: string;
  targetSkill: string;
}
