export type MasteryStatus = "weak" | "partial" | "strong";
export type MasteryTrend = "improving" | "declining" | "steady" | "not_available";

export interface ProfileSkill {
  skill: string;
  mastery_probability: number;
  mastery_status: MasteryStatus;
  previous_mastery_probability: number | null;
  trend: MasteryTrend;
  last_practiced_at: string | null;
}

export interface ProfileRecentSession {
  completed_at: string;
  skills: string[];
  questions_answered: number;
  correct_answers: number;
  incorrect_answers: number;
}

export interface StudentProfile {
  username: string;
  age: number | null;
  recent_session: ProfileRecentSession | null;
  weekly_summary: {
    window_days: 7;
    sessions: number;
    skills_practiced: number;
    questions_answered: number;
    correct_answers: number;
  };
  focus_next: {
    skill: string;
    reason: string;
    source: "student_model_learning_path";
  } | null;
  skills: ProfileSkill[];
  total_practiced_skills: number;
}
