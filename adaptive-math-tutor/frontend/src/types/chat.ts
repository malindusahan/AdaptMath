import type {
  AssessmentQuestion,
  EvaluationResult,
  StudentAnswerPayload,
  TutorSessionResponse,
} from "./tutor";

export type ChatMessage =
  | {
      id: string;
      type: "user";
      createdAt: string;
      content: string;
      threadId?: string;
    }
  | {
      id: string;
      type: "assistant";
      createdAt: string;
      content: string;
      threadId: string;
      reteachRound: number;
      turnCount: number;
    }
  | {
      id: string;
      type: "assessment";
      createdAt: string;
      threadId: string;
      reteachRound: number;
      message?: string | null;
      questions: AssessmentQuestion[];
      submitted: boolean;
      answers?: StudentAnswerPayload[];
    }
  | {
      id: string;
      type: "evaluation";
      createdAt: string;
      threadId: string;
      reteachRound: number;
      evaluation: EvaluationResult;
    }
  | {
      id: string;
      type: "recovery";
      createdAt: string;
      threadId: string;
      content: string;
    }
  | {
      id: string;
      type: "completion";
      createdAt: string;
      threadId: string;
      content: string;
    };

export interface ChatConversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessage[];
  activeThreadId: string | null;
  session: TutorSessionResponse | null;
}
