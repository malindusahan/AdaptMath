import {
  Calculator,
  FunctionSquare,
  Shapes,
  Sparkles,
} from "lucide-react";

interface EmptyChatProps {
  onExample: (question: string) => void;
}

const examples = [
  {
    icon: FunctionSquare,
    color: "from-violet-500 to-indigo-600",
    label: "Solve an equation",
    question: "Solve 3x + 5 = 20 and explain each step.",
  },
  {
    icon: Shapes,
    color: "from-cyan-500 to-blue-600",
    label: "Geometry problem",
    question: "A rectangle has a perimeter of 50 cm. Its length is 5 cm more than twice its width. Find its dimensions.",
  },
  {
    icon: Calculator,
    color: "from-amber-400 to-orange-500",
    label: "Percentages",
    question: "A school bag costs Rs. 4,500 and has a 20% discount. Find the discount and final price.",
  },
];

export function EmptyChat({ onExample }: EmptyChatProps) {
  return (
    <div className="mx-auto flex min-h-full w-full max-w-4xl flex-1 flex-col items-center justify-center px-4 pb-24 pt-12 text-center sm:px-8">
      <div className="flex items-center gap-2 rounded-full border border-indigo-100 bg-indigo-50 px-3 py-1.5 text-xs font-bold text-indigo-700 dark:border-indigo-500/20 dark:bg-indigo-500/10 dark:text-indigo-300">
        <Sparkles className="h-3.5 w-3.5" aria-hidden="true" />
        Your personal maths coach
      </div>
      <h1 className="mt-5 bg-gradient-to-r from-slate-950 via-indigo-800 to-violet-700 bg-clip-text text-3xl font-extrabold tracking-tight text-transparent dark:from-white dark:via-indigo-200 dark:to-violet-300 sm:text-4xl">
        What will we solve today?
      </h1>
      <p className="mt-3 max-w-xl text-sm leading-6 text-slate-500 dark:text-slate-400 sm:text-base">
        Ask any maths question. I’ll explain it step by step and adjust the lesson to match your pace.
      </p>

      <div className="mt-9 grid w-full gap-3 sm:grid-cols-3">
        {examples.map(({ icon: Icon, color, label, question }) => (
          <button
            key={label}
            type="button"
            onClick={() => onExample(question)}
            className="group rounded-3xl border border-slate-200/80 bg-white p-4 text-left shadow-sm transition duration-200 hover:-translate-y-1 hover:border-indigo-200 hover:shadow-lg hover:shadow-indigo-900/5 dark:border-slate-800 dark:bg-slate-900 dark:hover:border-indigo-500/40 dark:hover:bg-slate-800/80"
          >
            <span className={`flex h-10 w-10 items-center justify-center rounded-2xl bg-gradient-to-br ${color} text-white shadow-sm transition group-hover:scale-105`}>
              <Icon className="h-5 w-5" aria-hidden="true" />
            </span>
            <div className="mt-4 text-sm font-bold text-slate-900 dark:text-slate-100">{label}</div>
            <div className="mt-1.5 line-clamp-2 text-xs leading-5 text-slate-500 dark:text-slate-400">{question}</div>
          </button>
        ))}
      </div>
    </div>
  );
}
