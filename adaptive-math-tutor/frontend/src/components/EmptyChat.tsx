import { Calculator, FunctionSquare, Shapes, Sparkles } from "lucide-react";

interface EmptyChatProps {
  onExample: (question: string) => void;
}

const examples = [
  {
    icon: FunctionSquare,
    label: "Solve an equation",
    question: "Solve 3x + 5 = 20 and explain each step.",
  },
  {
    icon: Shapes,
    label: "Geometry problem",
    question: "A rectangle has a perimeter of 50 cm. Its length is 5 cm more than twice its width. Find its dimensions.",
  },
  {
    icon: Calculator,
    label: "Percentages",
    question: "A school bag costs Rs. 4,500 and has a 20% discount. Find the discount and final price.",
  },
];

export function EmptyChat({ onExample }: EmptyChatProps) {
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col items-center justify-center px-4 pb-24 pt-16 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-full bg-zinc-900 text-white shadow-sm">
        <Sparkles className="h-5 w-5" aria-hidden="true" />
      </div>
      <h1 className="mt-5 text-2xl font-semibold tracking-tight text-zinc-950 sm:text-3xl">How can I help with your maths?</h1>
      <p className="mt-2 max-w-lg text-sm leading-6 text-zinc-500">
        Ask a question naturally. AdaptMath will decide how much explanation, planning, and checking you need behind the scenes.
      </p>

      <div className="mt-8 grid w-full gap-2 sm:grid-cols-3">
        {examples.map(({ icon: Icon, label, question }) => (
          <button
            key={label}
            type="button"
            onClick={() => onExample(question)}
            className="rounded-2xl border border-zinc-200 bg-white p-4 text-left transition hover:bg-zinc-50"
          >
            <Icon className="h-5 w-5 text-zinc-600" aria-hidden="true" />
            <div className="mt-3 text-sm font-medium text-zinc-900">{label}</div>
            <div className="mt-1 line-clamp-2 text-xs leading-5 text-zinc-500">{question}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
