import { useEffect, useMemo, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  BookOpenCheck,
  CalendarDays,
  ChevronDown,
  ChevronUp,
  CircleAlert,
  RotateCw,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
  UserRound,
} from "lucide-react";

import { ApiError, getStudentProfile } from "../api/client";
import type { MasteryStatus, ProfileSkill, StudentProfile } from "../types/profile";
import { ThemeToggle } from "./ThemeToggle";

interface StudentProfilePageProps {
  onBack: () => void;
  onStartPracticing: (skill: string) => Promise<void>;
}

const statusLabels: Record<MasteryStatus, string> = {
  strong: "Strong",
  partial: "Developing",
  weak: "Needs practice",
};

const statusClasses: Record<MasteryStatus, string> = {
  strong: "bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300",
  partial: "bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300",
  weak: "bg-rose-50 text-rose-700 dark:bg-rose-500/10 dark:text-rose-300",
};

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(
    new Date(value),
  );
}

function Trend({ skill }: { skill: ProfileSkill }) {
  if (skill.trend === "improving") {
    return (
      <span className="flex items-center gap-1 text-emerald-700 dark:text-emerald-300">
        <TrendingUp className="h-3.5 w-3.5" /> Improving
      </span>
    );
  }
  if (skill.trend === "declining") {
    return (
      <span className="flex items-center gap-1 text-amber-700 dark:text-amber-300">
        <TrendingDown className="h-3.5 w-3.5" /> Review suggested
      </span>
    );
  }
  if (skill.trend === "steady") {
    return <span className="text-slate-500 dark:text-slate-400">Steady</span>;
  }
  return <span className="text-slate-400 dark:text-slate-500">First result</span>;
}

function LoadingProfile() {
  return (
    <div className="mx-auto grid w-full max-w-6xl animate-pulse gap-5 px-4 py-8 sm:px-6">
      <div className="h-48 rounded-[2rem] bg-slate-200/70 dark:bg-slate-800" />
      <div className="grid gap-5 md:grid-cols-2">
        <div className="h-48 rounded-3xl bg-slate-200/70 dark:bg-slate-800" />
        <div className="h-48 rounded-3xl bg-slate-200/70 dark:bg-slate-800" />
      </div>
      <div className="h-64 rounded-3xl bg-slate-200/70 dark:bg-slate-800" />
    </div>
  );
}

export function StudentProfilePage({
  onBack,
  onStartPracticing,
}: StudentProfilePageProps) {
  const [profile, setProfile] = useState<StudentProfile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | MasteryStatus>("all");
  const [expanded, setExpanded] = useState(false);
  const [preparingSkill, setPreparingSkill] = useState<string | null>(null);
  const [practiceError, setPracticeError] = useState<string | null>(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      setProfile(await getStudentProfile());
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught.detail
          : "Your learning profile could not be loaded.",
      );
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load();
  }, []);

  async function startPractice(skill: string) {
    if (preparingSkill) return;
    setPreparingSkill(skill);
    setPracticeError(null);
    try {
      await onStartPracticing(skill);
    } catch (caught) {
      setPracticeError(
        caught instanceof ApiError
          ? caught.detail
          : caught instanceof Error
            ? caught.message
            : "We couldn't prepare a practice problem right now.",
      );
    } finally {
      setPreparingSkill(null);
    }
  }

  const filtered = useMemo(
    () =>
      profile?.skills.filter(
        (skill) => filter === "all" || skill.mastery_status === filter,
      ) ?? [],
    [filter, profile],
  );
  const visibleSkills = expanded ? filtered : filtered.slice(0, 6);

  return (
    <main className="h-screen overflow-y-auto bg-[#f8faff] text-slate-900 transition-colors dark:bg-slate-950 dark:text-slate-100">
      <header className="sticky top-0 z-20 border-b border-indigo-100 bg-white/85 backdrop-blur-xl dark:border-slate-800 dark:bg-slate-950/85">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
          <button
            type="button"
            onClick={onBack}
            className="flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-bold text-slate-700 transition hover:bg-indigo-50 hover:text-indigo-700 dark:text-slate-200 dark:hover:bg-slate-800 dark:hover:text-indigo-300"
          >
            <ArrowLeft className="h-4 w-4" />
            <span className="hidden sm:inline">Back to Tutor</span>
            <span className="sm:hidden">Back</span>
          </button>
          <div className="text-sm font-bold">My learning</div>
          <ThemeToggle />
        </div>
      </header>

      {loading ? (
        <LoadingProfile />
      ) : error ? (
        <div className="mx-auto flex max-w-xl flex-col items-center px-5 py-24 text-center">
          <CircleAlert className="h-9 w-9 text-rose-600 dark:text-rose-400" />
          <h1 className="mt-4 text-xl font-bold">Profile unavailable</h1>
          <p className="mt-2 text-sm leading-6 text-slate-500 dark:text-slate-400">{error}</p>
          <button
            type="button"
            onClick={() => void load()}
            className="mt-5 flex items-center gap-2 rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-bold text-white hover:bg-indigo-500"
          >
            <RotateCw className="h-4 w-4" /> Try again
          </button>
        </div>
      ) : (
        profile && (
          <div className="mx-auto grid w-full max-w-6xl gap-5 px-4 py-6 sm:px-6 sm:py-8">
            <section className="relative overflow-hidden rounded-[2rem] bg-gradient-to-br from-indigo-600 via-violet-600 to-fuchsia-500 p-6 text-white shadow-xl shadow-indigo-900/10 sm:p-8">
              <div className="absolute -right-16 -top-20 h-64 w-64 rounded-full border-[40px] border-white/10" />
              <div className="absolute -bottom-24 right-1/3 h-52 w-52 rounded-full bg-cyan-300/20 blur-2xl" />
              <div className="relative flex flex-col gap-5 sm:flex-row sm:items-center">
                <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl bg-white/20 text-2xl font-extrabold shadow-inner ring-1 ring-white/30 backdrop-blur">
                  {profile.username.slice(0, 1).toUpperCase() || <UserRound />}
                </div>
                <div>
                  <p className="text-xs font-bold uppercase tracking-[0.2em] text-indigo-100">
                    Your learning journey
                  </p>
                  <h1 className="mt-2 text-3xl font-extrabold tracking-tight sm:text-4xl">
                    Hi {profile.username}!
                  </h1>
                  <p className="mt-2 max-w-xl text-sm leading-6 text-indigo-50 sm:text-base">
                    See what you’ve mastered and choose your next challenge.
                    {profile.age ? ` You’re learning at age ${profile.age}.` : ""}
                  </p>
                </div>
              </div>
            </section>

            <div className="grid gap-5 md:grid-cols-2">
              <DashboardCard icon={BookOpenCheck} title="Recent session">
                {profile.recent_session ? (
                  <div className="mt-5">
                    <p className="text-xs font-medium text-slate-500 dark:text-slate-400">
                      {formatDate(profile.recent_session.completed_at)}
                    </p>
                    <h2 className="mt-2 text-lg font-bold">
                      {profile.recent_session.skills.join(", ") || "Maths practice"}
                    </h2>
                    <div className="mt-5 grid grid-cols-3 gap-3">
                      <Metric value={profile.recent_session.questions_answered} label="Answered" />
                      <Metric value={profile.recent_session.correct_answers} label="Correct" />
                      <Metric value={profile.recent_session.incorrect_answers} label="Review" />
                    </div>
                  </div>
                ) : (
                  <Empty text="Complete your first lesson to start building your learning profile." />
                )}
              </DashboardCard>

              <DashboardCard icon={CalendarDays} title="Last 7 days">
                <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4 md:grid-cols-2 lg:grid-cols-4">
                  <Metric value={profile.weekly_summary.sessions} label="Sessions" />
                  <Metric value={profile.weekly_summary.skills_practiced} label="Skills" />
                  <Metric value={profile.weekly_summary.questions_answered} label="Answered" />
                  <Metric value={profile.weekly_summary.correct_answers} label="Correct" />
                </div>
                {profile.weekly_summary.sessions === 0 && (
                  <p className="mt-4 text-xs text-slate-500 dark:text-slate-400">
                    No completed sessions in the last seven days.
                  </p>
                )}
              </DashboardCard>
            </div>

            <section className="overflow-hidden rounded-3xl border border-indigo-200 bg-gradient-to-br from-indigo-50 via-white to-violet-50 p-6 shadow-sm dark:border-indigo-500/20 dark:from-indigo-950/40 dark:via-slate-900 dark:to-violet-950/30 sm:p-7">
              <div className="flex flex-col justify-between gap-6 sm:flex-row sm:items-end">
                <div className="max-w-2xl">
                  <div className="flex items-center gap-2 text-xs font-extrabold uppercase tracking-[0.2em] text-indigo-700 dark:text-indigo-300">
                    <Target className="h-4 w-4" /> Focus next
                  </div>
                  {profile.focus_next ? (
                    <>
                      <h2 className="mt-3 text-2xl font-extrabold tracking-tight">
                        {profile.focus_next.skill}
                      </h2>
                      <p className="mt-2 text-sm leading-6 text-slate-600 dark:text-slate-300">
                        {profile.focus_next.reason}
                      </p>
                    </>
                  ) : (
                    <>
                      <h2 className="mt-3 text-xl font-bold">No recommendation yet</h2>
                      <p className="mt-2 text-sm text-slate-600 dark:text-slate-300">
                        Complete a lesson to build enough evidence for your learning path.
                      </p>
                    </>
                  )}
                </div>
                {profile.focus_next && (
                  <button
                    type="button"
                    disabled={Boolean(preparingSkill)}
                    onClick={() => void startPractice(profile.focus_next!.skill)}
                    className="flex shrink-0 items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-indigo-600 to-violet-600 px-5 py-3 text-sm font-bold text-white shadow-lg shadow-indigo-500/20 transition hover:-translate-y-0.5 disabled:translate-y-0 disabled:cursor-wait disabled:opacity-60"
                  >
                    {preparingSkill ? (
                      <>
                        <RotateCw className="h-4 w-4 animate-spin" /> Preparing practice…
                      </>
                    ) : (
                      <>
                        Start practicing <ArrowRight className="h-4 w-4" />
                      </>
                    )}
                  </button>
                )}
              </div>
              {practiceError && profile.focus_next && (
                <div className="mt-5 rounded-2xl border border-rose-200 bg-white/80 p-4 dark:border-rose-500/20 dark:bg-rose-500/10">
                  <div className="flex gap-2 text-sm text-rose-700 dark:text-rose-300">
                    <CircleAlert className="mt-0.5 h-4 w-4 shrink-0" />
                    <p>{practiceError}</p>
                  </div>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => void startPractice(profile.focus_next!.skill)}
                      className="flex items-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-xs font-bold text-white"
                    >
                      <RotateCw className="h-3.5 w-3.5" /> Try again
                    </button>
                    <button
                      type="button"
                      onClick={onBack}
                      className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs font-bold text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200"
                    >
                      Choose your own problem
                    </button>
                  </div>
                </div>
              )}
            </section>

            <section className="rounded-3xl border border-slate-200/80 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900 sm:p-6">
              <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <div className="flex items-center gap-2 text-sm font-bold">
                    <Sparkles className="h-4.5 w-4.5 text-indigo-600 dark:text-indigo-300" />
                    Your skills
                  </div>
                  <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                    Your progress updates as you complete lessons.
                  </p>
                </div>
                {profile.skills.length > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {(["all", "strong", "partial", "weak"] as const).map((value) => (
                      <button
                        key={value}
                        type="button"
                        onClick={() => {
                          setFilter(value);
                          setExpanded(false);
                        }}
                        className={`rounded-full px-3 py-1.5 text-xs font-bold transition ${
                          filter === value
                            ? "bg-indigo-600 text-white"
                            : "bg-slate-100 text-slate-600 hover:bg-indigo-50 hover:text-indigo-700 dark:bg-slate-800 dark:text-slate-300 dark:hover:bg-slate-700"
                        }`}
                      >
                        {value === "all" ? "All" : statusLabels[value]}
                      </button>
                    ))}
                  </div>
                )}
              </div>
              {profile.skills.length === 0 ? (
                <Empty text="Your practiced skills and mastery will appear here after your first completed lesson." />
              ) : (
                <div className="mt-5 divide-y divide-slate-100 dark:divide-slate-800">
                  {visibleSkills.map((skill) => (
                    <SkillRow key={skill.skill} skill={skill} />
                  ))}
                </div>
              )}
              {filtered.length > 6 && (
                <button
                  type="button"
                  onClick={() => setExpanded((value) => !value)}
                  className="mx-auto mt-4 flex items-center gap-1.5 rounded-xl px-3 py-2 text-sm font-bold text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
                >
                  {expanded ? (
                    <><ChevronUp className="h-4 w-4" /> Show less</>
                  ) : (
                    <><ChevronDown className="h-4 w-4" /> View all {filtered.length} skills</>
                  )}
                </button>
              )}
            </section>
          </div>
        )
      )}
    </main>
  );
}

function DashboardCard({
  icon: Icon,
  title,
  children,
}: {
  icon: typeof BookOpenCheck;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-3xl border border-slate-200/80 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="flex items-center gap-2 text-sm font-bold">
        <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-indigo-50 text-indigo-600 dark:bg-indigo-500/10 dark:text-indigo-300">
          <Icon className="h-4.5 w-4.5" />
        </span>
        {title}
      </div>
      {children}
    </section>
  );
}

function Metric({ value, label }: { value: number; label: string }) {
  return (
    <div className="rounded-2xl bg-slate-50 px-3 py-3 dark:bg-slate-800/80">
      <div className="text-xl font-extrabold tabular-nums text-indigo-700 dark:text-indigo-300">
        {value}
      </div>
      <div className="mt-0.5 text-[11px] font-medium text-slate-500 dark:text-slate-400">
        {label}
      </div>
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return (
    <div className="mt-5 rounded-2xl border border-dashed border-slate-200 bg-slate-50 px-4 py-6 text-sm leading-6 text-slate-500 dark:border-slate-700 dark:bg-slate-800/50 dark:text-slate-400">
      {text}
    </div>
  );
}

function SkillRow({ skill }: { skill: ProfileSkill }) {
  const percent = Math.round(skill.mastery_probability * 100);
  return (
    <div className="grid gap-3 py-4 sm:grid-cols-[minmax(0,1fr)_minmax(180px,0.7fr)_auto] sm:items-center">
      <div className="min-w-0">
        <div className="truncate text-sm font-bold">{skill.skill}</div>
        <div className="mt-1 text-xs"><Trend skill={skill} /></div>
      </div>
      <div className="flex items-center gap-3">
        <div className="h-2.5 flex-1 overflow-hidden rounded-full bg-slate-100 dark:bg-slate-800">
          <div
            className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-violet-500"
            style={{ width: `${percent}%` }}
          />
        </div>
        <span className="w-10 text-right text-sm font-extrabold tabular-nums">{percent}%</span>
      </div>
      <span className={`w-fit rounded-full px-2.5 py-1 text-[11px] font-bold ${statusClasses[skill.mastery_status]}`}>
        {statusLabels[skill.mastery_status]}
      </span>
    </div>
  );
}
