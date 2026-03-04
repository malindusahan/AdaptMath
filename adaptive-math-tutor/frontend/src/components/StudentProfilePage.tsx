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
  Settings2,
  Sparkles,
  Target,
  TrendingDown,
  TrendingUp,
  UserRound,
} from "lucide-react";

import { ApiError, getStudentProfile } from "../api/client";
import type { MasteryStatus, ProfileSkill, StudentProfile } from "../types/profile";

interface StudentProfilePageProps {
  onBack: () => void;
  onOpenSettings: () => void;
  onStartPracticing: (skill: string) => Promise<void>;
}

const statusLabels: Record<MasteryStatus, string> = {
  strong: "Strong",
  partial: "Developing",
  weak: "Needs practice",
};

const statusClasses: Record<MasteryStatus, string> = {
  strong: "bg-emerald-50 text-emerald-700",
  partial: "bg-amber-50 text-amber-700",
  weak: "bg-rose-50 text-rose-700",
};

function formatDate(value: string): string {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(new Date(value));
}

function Trend({ skill }: { skill: ProfileSkill }) {
  if (skill.trend === "improving") return <span className="flex items-center gap-1 text-emerald-700"><TrendingUp className="h-3.5 w-3.5" />Improving</span>;
  if (skill.trend === "declining") return <span className="flex items-center gap-1 text-amber-700"><TrendingDown className="h-3.5 w-3.5" />Review suggested</span>;
  if (skill.trend === "steady") return <span className="text-zinc-500">Steady</span>;
  return <span className="text-zinc-400">First result</span>;
}

function LoadingProfile() {
  return <div className="mx-auto grid w-full max-w-6xl animate-pulse gap-5 px-4 py-8 sm:px-6"><div className="h-44 rounded-3xl bg-zinc-100"/><div className="grid gap-5 md:grid-cols-2"><div className="h-48 rounded-2xl bg-zinc-100"/><div className="h-48 rounded-2xl bg-zinc-100"/></div><div className="h-64 rounded-2xl bg-zinc-100"/></div>;
}

export function StudentProfilePage({ onBack, onOpenSettings, onStartPracticing }: StudentProfilePageProps) {
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
      setError(caught instanceof ApiError ? caught.detail : "Your learning profile could not be loaded.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { void load(); }, []);

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

  const filtered = useMemo(() => profile?.skills.filter((skill) => filter === "all" || skill.mastery_status === filter) ?? [], [filter, profile]);
  const visibleSkills = expanded ? filtered : filtered.slice(0, 6);

  return (
    <main className="h-screen overflow-y-auto bg-[#f7f7f8] text-zinc-900">
      <header className="sticky top-0 z-20 border-b border-zinc-200/80 bg-white/90 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
          <button type="button" onClick={onBack} className="flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium text-zinc-700 hover:bg-zinc-100"><ArrowLeft className="h-4 w-4"/>Back to Tutor</button>
          <div className="text-sm font-semibold">Student Profile</div>
          <button type="button" onClick={onOpenSettings} className="rounded-xl p-2.5 text-zinc-600 hover:bg-zinc-100" aria-label="Open learner context settings"><Settings2 className="h-4.5 w-4.5"/></button>
        </div>
      </header>

      {loading ? <LoadingProfile/> : error ? (
        <div className="mx-auto flex max-w-xl flex-col items-center px-5 py-24 text-center"><CircleAlert className="h-8 w-8 text-rose-600"/><h1 className="mt-4 text-xl font-semibold">Profile unavailable</h1><p className="mt-2 text-sm leading-6 text-zinc-500">{error}</p><button type="button" onClick={() => void load()} className="mt-5 flex items-center gap-2 rounded-xl bg-zinc-900 px-4 py-2.5 text-sm font-medium text-white"><RotateCw className="h-4 w-4"/>Try again</button></div>
      ) : profile && (
        <div className="mx-auto grid w-full max-w-6xl gap-5 px-4 py-6 sm:px-6 sm:py-8">
          <section className="relative overflow-hidden rounded-3xl border border-zinc-200 bg-white p-6 shadow-sm sm:p-8">
            <div className="absolute -right-20 -top-24 h-64 w-64 rounded-full bg-indigo-100/60 blur-3xl"/>
            <div className="relative flex flex-col gap-5 sm:flex-row sm:items-center">
              <div className="flex h-16 w-16 shrink-0 items-center justify-center rounded-2xl bg-zinc-900 text-xl font-semibold text-white">{profile.username.slice(0, 1).toUpperCase() || <UserRound/>}</div>
              <div><p className="text-xs font-semibold uppercase tracking-[0.18em] text-zinc-500">Learning overview</p><h1 className="mt-2 text-3xl font-semibold tracking-tight">Hi {profile.username}</h1><p className="mt-2 text-sm leading-6 text-zinc-500">Here’s how your math learning is progressing.{profile.age ? ` Age ${profile.age}.` : ""}</p></div>
            </div>
          </section>

          <div className="grid gap-5 md:grid-cols-2">
            <section className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm"><div className="flex items-center gap-2 text-sm font-semibold"><BookOpenCheck className="h-4.5 w-4.5 text-indigo-600"/>Recent session</div>{profile.recent_session ? <div className="mt-5"><p className="text-xs text-zinc-500">{formatDate(profile.recent_session.completed_at)}</p><h2 className="mt-2 text-lg font-semibold">{profile.recent_session.skills.join(", ") || "Math practice"}</h2><div className="mt-5 grid grid-cols-3 gap-3"><Metric value={profile.recent_session.questions_answered} label="Answered"/><Metric value={profile.recent_session.correct_answers} label="Correct"/><Metric value={profile.recent_session.incorrect_answers} label="Review"/></div></div> : <Empty text="Complete your first lesson to start building your learning profile."/>}</section>
            <section className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm"><div className="flex items-center gap-2 text-sm font-semibold"><CalendarDays className="h-4.5 w-4.5 text-indigo-600"/>Last 7 days</div><div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4 md:grid-cols-2 lg:grid-cols-4"><Metric value={profile.weekly_summary.sessions} label="Sessions"/><Metric value={profile.weekly_summary.skills_practiced} label="Skills"/><Metric value={profile.weekly_summary.questions_answered} label="Answered"/><Metric value={profile.weekly_summary.correct_answers} label="Correct"/></div>{profile.weekly_summary.sessions === 0 && <p className="mt-4 text-xs text-zinc-500">No completed sessions in the last seven days.</p>}</section>
          </div>

          <section className="overflow-hidden rounded-2xl border border-indigo-200 bg-gradient-to-br from-indigo-50 via-white to-violet-50 p-6 shadow-sm sm:p-7">
            <div className="flex flex-col justify-between gap-6 sm:flex-row sm:items-end">
              <div className="max-w-2xl">
                <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.2em] text-indigo-700"><Target className="h-4 w-4"/>Focus next</div>
                {profile.focus_next ? <><h2 className="mt-3 text-2xl font-semibold tracking-tight">{profile.focus_next.skill}</h2><p className="mt-2 text-sm leading-6 text-zinc-600">{profile.focus_next.reason}</p></> : <><h2 className="mt-3 text-xl font-semibold">No recommendation yet</h2><p className="mt-2 text-sm text-zinc-600">Complete a lesson to build enough evidence for your learning path.</p></>}
              </div>
              {profile.focus_next && <button type="button" disabled={Boolean(preparingSkill)} onClick={() => void startPractice(profile.focus_next!.skill)} className="flex shrink-0 items-center justify-center gap-2 rounded-xl bg-zinc-900 px-5 py-3 text-sm font-medium text-white shadow-sm hover:bg-zinc-800 disabled:cursor-wait disabled:bg-zinc-600">{preparingSkill ? <><RotateCw className="h-4 w-4 animate-spin"/>Preparing practice...</> : <>Start practicing<ArrowRight className="h-4 w-4"/></>}</button>}
            </div>
            {practiceError && profile.focus_next && <div className="mt-5 rounded-xl border border-rose-200 bg-white/80 p-4"><div className="flex gap-2 text-sm text-rose-700"><CircleAlert className="mt-0.5 h-4 w-4 shrink-0"/><p>{practiceError}</p></div><div className="mt-3 flex flex-wrap gap-2"><button type="button" onClick={() => void startPractice(profile.focus_next!.skill)} className="flex items-center gap-2 rounded-lg bg-zinc-900 px-3 py-2 text-xs font-medium text-white"><RotateCw className="h-3.5 w-3.5"/>Try again</button><button type="button" onClick={onBack} className="rounded-lg border border-zinc-200 bg-white px-3 py-2 text-xs font-medium text-zinc-700 hover:bg-zinc-50">Choose your own problem</button></div></div>}
          </section>

          <section className="rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm sm:p-6"><div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between"><div><div className="flex items-center gap-2 text-sm font-semibold"><Sparkles className="h-4.5 w-4.5 text-indigo-600"/>Your skills</div><p className="mt-1 text-xs text-zinc-500">Mastery comes from your BKT learning record.</p></div>{profile.skills.length > 0 && <div className="flex flex-wrap gap-1.5">{(["all","strong","partial","weak"] as const).map((value) => <button key={value} type="button" onClick={() => {setFilter(value);setExpanded(false);}} className={`rounded-full px-3 py-1.5 text-xs font-medium ${filter === value ? "bg-zinc-900 text-white" : "bg-zinc-100 text-zinc-600 hover:bg-zinc-200"}`}>{value === "all" ? "All" : statusLabels[value]}</button>)}</div>}</div>
            {profile.skills.length === 0 ? <Empty text="Your practiced skills and mastery will appear here after your first completed lesson."/> : <div className="mt-5 divide-y divide-zinc-100">{visibleSkills.map((skill) => <SkillRow key={skill.skill} skill={skill}/>)}</div>}
            {filtered.length > 6 && <button type="button" onClick={() => setExpanded((value) => !value)} className="mx-auto mt-4 flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium text-zinc-600 hover:bg-zinc-100">{expanded ? <><ChevronUp className="h-4 w-4"/>Show less</> : <><ChevronDown className="h-4 w-4"/>View all {filtered.length} skills</>}</button>}
          </section>
        </div>
      )}
    </main>
  );
}

function Metric({ value, label }: { value: number; label: string }) { return <div className="rounded-xl bg-zinc-50 px-3 py-3"><div className="text-xl font-semibold tabular-nums">{value}</div><div className="mt-0.5 text-[11px] text-zinc-500">{label}</div></div>; }
function Empty({ text }: { text: string }) { return <div className="mt-5 rounded-xl border border-dashed border-zinc-200 bg-zinc-50 px-4 py-6 text-sm leading-6 text-zinc-500">{text}</div>; }
function SkillRow({ skill }: { skill: ProfileSkill }) { const percent=Math.round(skill.mastery_probability*100); return <div className="grid gap-3 py-4 sm:grid-cols-[minmax(0,1fr)_minmax(180px,0.7fr)_auto] sm:items-center"><div className="min-w-0"><div className="truncate text-sm font-medium">{skill.skill}</div><div className="mt-1 text-xs"><Trend skill={skill}/></div></div><div className="flex items-center gap-3"><div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-100"><div className="h-full rounded-full bg-indigo-600" style={{width:`${percent}%`}}/></div><span className="w-10 text-right text-sm font-semibold tabular-nums">{percent}%</span></div><span className={`w-fit rounded-full px-2.5 py-1 text-[11px] font-semibold ${statusClasses[skill.mastery_status]}`}>{statusLabels[skill.mastery_status]}</span></div>; }
