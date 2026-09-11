import type { ChatMessage } from "../types/chat";
import type { TutorDialogueTurn } from "../types/tutor";

const MAX_CONTINUATION_TURNS = 100;

export function originalLessonQuestion(messages: ChatMessage[], fallback: string): string {
  const firstQuestion = messages.find((message) => message.type === "user");
  return firstQuestion?.content ?? fallback;
}

export function continuationHistory(messages: ChatMessage[]): TutorDialogueTurn[] {
  let skippedOriginalQuestion = false;
  const turns: TutorDialogueTurn[] = [];

  for (const message of messages) {
    if (message.type === "user") {
      if (!skippedOriginalQuestion) {
        skippedOriginalQuestion = true;
        continue;
      }
      turns.push({ role: "student", content: message.content });
      continue;
    }

    if (message.type === "assistant") {
      turns.push({ role: "teacher", content: message.content });
      continue;
    }

    if (message.type === "assessment" && message.submitted && message.answers) {
      const answerById = new Map(
        message.answers.map((answer) => [answer.question_id, answer.answer]),
      );
      for (const question of message.questions) {
        const answer = answerById.get(question.question_id);
        if (!answer) continue;
        turns.push({ role: "teacher", content: `Assessment: ${question.question}` });
        turns.push({ role: "student", content: answer });
      }
    }
  }

  return turns.slice(-MAX_CONTINUATION_TURNS);
}

export function messagesForPausedLesson(messages: ChatMessage[]): ChatMessage[] {
  return messages.filter(
    (message) =>
      message.type !== "recovery" &&
      !(message.type === "assessment" && !message.submitted),
  );
}
