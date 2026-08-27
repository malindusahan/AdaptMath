import React, { useState } from 'react'
import PageHeader from '../components/PageHeader'
import SectionCard from '../components/SectionCard'
import EmptyState from '../components/EmptyState'
import StatusBadge, { BadgeVariant } from '../components/StatusBadge'
import LoadingSpinner from '../components/LoadingSpinner'
import { getQuestionContext, submitMemoryUpdate } from '../api/memoryApi'
import {
  AssessmentQuestionInput,
  MemoryUpdateResponse,
  QuestionContextResponse,
} from '../types/api'

interface QuestionFormItem {
  id: string
  question: string
  student_answer: string
  expected_answer: string
  is_correct: boolean
  identified_error: string
  attempt_count: string
  hint_count: string
  response_time_ms: string
}

export const StudentPage: React.FC = () => {
  // Question Analysis State
  const [studentId, setStudentId] = useState('demo_student_01')
  const [sessionId, setSessionId] = useState('demo_session_01')
  const [questionText, setQuestionText] = useState('How do I solve the linear equation 3x + 9 = 24?')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [analysisResult, setAnalysisResult] = useState<QuestionContextResponse | null>(null)

  // Assessment Update State
  const [assessmentTopic, setAssessmentTopic] = useState('Algebra')
  const [assessmentSubtopic, setAssessmentSubtopic] = useState('Linear equations')
  const [questions, setQuestions] = useState<QuestionFormItem[]>([
    {
      id: 'q_1',
      question: 'Solve for x: 3x = 15',
      student_answer: '5',
      expected_answer: '5',
      is_correct: true,
      identified_error: '',
      attempt_count: '1',
      hint_count: '0',
      response_time_ms: '3800',
    },
  ])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  const [updateResult, setUpdateResult] = useState<MemoryUpdateResponse | null>(null)

  const handleAnalyze = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    if (!studentId.trim() || !questionText.trim()) return

    setIsAnalyzing(true)
    setAnalysisError(null)

    try {
      const data = await getQuestionContext({
        student_id: studentId.trim(),
        session_id: sessionId.trim() || 'default_session',
        question: questionText.trim(),
      })
      setAnalysisResult(data)
    } catch (err: unknown) {
      setAnalysisResult(null)
      const errorObj = err as { response?: { data?: { message?: string } }; message?: string }
      const message =
        errorObj?.response?.data?.message ||
        errorObj?.message ||
        'Unable to process question context. Ensure the memory service is running.'
      setAnalysisError(message)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleAddQuestion = () => {
    setQuestions([
      ...questions,
      {
        id: `q_${Date.now()}`,
        question: `Solve for x: 2x + 4 = 10`,
        student_answer: '3',
        expected_answer: '3',
        is_correct: true,
        identified_error: '',
        attempt_count: '1',
        hint_count: '0',
        response_time_ms: '4200',
      },
    ])
  }

  const handleRemoveQuestion = (id: string) => {
    if (questions.length <= 1) return
    setQuestions(questions.filter((q) => q.id !== id))
  }

  const handleQuestionChange = (
    id: string,
    field: keyof QuestionFormItem,
    value: string | boolean
  ) => {
    setQuestions(
      questions.map((q) => (q.id === id ? { ...q, [field]: value } : q))
    )
  }

  const handleSubmitAssessment = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!studentId.trim() || !assessmentTopic.trim()) return

    setIsSubmitting(true)
    setSubmitError(null)

    try {
      const formattedQuestions: AssessmentQuestionInput[] = questions.map((q, idx) => ({
        question_id: `q_${idx + 1}_${Date.now()}`,
        question: q.question.trim() || undefined,
        student_answer: q.student_answer.trim() || undefined,
        expected_answer: q.expected_answer.trim() || undefined,
        is_correct: q.is_correct,
        identified_error: q.identified_error.trim() || null,
        attempt_count: q.attempt_count.trim() ? parseInt(q.attempt_count.trim(), 10) : null,
        hint_count: q.hint_count.trim() ? parseInt(q.hint_count.trim(), 10) : null,
        response_time_ms: q.response_time_ms.trim() ? parseFloat(q.response_time_ms.trim()) : null,
      }))

      const identifiedErrors = questions
        .map((q) => q.identified_error.trim())
        .filter(Boolean)

      const payload = {
        student_id: studentId.trim(),
        topic: assessmentTopic.trim(),
        subtopic: assessmentSubtopic.trim() || null,
        assessment_questions: formattedQuestions,
        identified_errors: identifiedErrors,
        overall_feedback: 'Assessment submitted from React demo.',
      }

      const response = await submitMemoryUpdate(payload)
      setUpdateResult(response)

      // Automatically refresh context to show latest updated cognitive state
      if (questionText.trim()) {
        await handleAnalyze()
      }
    } catch (err: unknown) {
      const errorObj = err as { response?: { data?: { message?: string } }; message?: string }
      const message =
        errorObj?.response?.data?.message ||
        errorObj?.message ||
        'Failed to record assessment update.'
      setSubmitError(message)
    } finally {
      setIsSubmitting(false)
    }
  }

  const getLearningStateVariant = (state?: string): BadgeVariant => {
    switch (state) {
      case 'STRONG':
        return 'success'
      case 'DEVELOPING':
        return 'warning'
      case 'NEEDS_SUPPORT':
        return 'error'
      default:
        return 'neutral'
    }
  }

  const topicAbstained =
    analysisResult?.topic?.needs_review ||
    !analysisResult?.topic?.display_name ||
    (analysisResult?.topic?.confidence ?? 0) < 0.5

  return (
    <div className="space-y-8">
      <PageHeader
        title="Student Demo"
        description="Interact with the autonomous question parser and submit evaluated assessments to observe dynamic cognitive memory updates in real time."
        badge={<StatusBadge label="Interactive Demo" variant="indigo" size="sm" />}
      />

      {/* SECTION 1: QUESTION ANALYSIS FORM */}
      <SectionCard
        title="1. Question Analysis & Memory Context"
        subtitle="Simulates a student querying the memory system with a natural language mathematics question."
      >
        <form onSubmit={handleAnalyze} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label
                htmlFor="student-id-input"
                className="block text-xs font-semibold text-slate-700 mb-1"
              >
                Student ID <span className="text-rose-500">*</span>
              </label>
              <input
                id="student-id-input"
                type="text"
                value={studentId}
                onChange={(e) => setStudentId(e.target.value)}
                placeholder="e.g. demo_student_01"
                className="w-full px-3 py-2 text-xs rounded-md border border-slate-300 bg-white shadow-xs focus:outline-none focus:ring-2 focus:ring-slate-900 font-mono"
                required
              />
            </div>
            <div>
              <label
                htmlFor="session-id-input"
                className="block text-xs font-semibold text-slate-700 mb-1"
              >
                Session ID <span className="text-slate-400 font-normal">(Optional)</span>
              </label>
              <input
                id="session-id-input"
                type="text"
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
                placeholder="e.g. demo_session_01"
                className="w-full px-3 py-2 text-xs rounded-md border border-slate-300 bg-white shadow-xs focus:outline-none focus:ring-2 focus:ring-slate-900 font-mono"
              />
            </div>
          </div>

          <div>
            <label
              htmlFor="question-textarea"
              className="block text-xs font-semibold text-slate-700 mb-1"
            >
              Ask a Mathematics Question <span className="text-rose-500">*</span>
            </label>
            <textarea
              id="question-textarea"
              rows={2}
              value={questionText}
              onChange={(e) => setQuestionText(e.target.value)}
              placeholder="Enter a mathematics question, e.g. 'Solve for x: 3x + 9 = 24' or 'How do I read a Box and Whisker plot?'"
              className="w-full px-3 py-2 text-xs rounded-md border border-slate-300 bg-white shadow-xs focus:outline-none focus:ring-2 focus:ring-slate-900 resize-y font-sans"
              required
            />
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
            <span className="text-[11px] text-slate-500">
              Dispatches question to the fine-tuned MiniLM topic extractor and fetches epistemic context.
            </span>
            <button
              type="submit"
              disabled={isAnalyzing}
              className="inline-flex items-center justify-center px-4 py-2 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 disabled:bg-slate-400 rounded-md shadow-xs transition-colors focus:outline-none focus:ring-2 focus:ring-slate-900 shrink-0"
            >
              {isAnalyzing ? 'Analyzing Question...' : 'Analyze Question'}
            </button>
          </div>
        </form>
      </SectionCard>

      {/* Analysis Error Message */}
      {analysisError && (
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start gap-2.5">
          <svg className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <span className="font-semibold block">Analysis Request Failed</span>
            <span>{analysisError}</span>
          </div>
        </div>
      )}

      {/* Analysis Loading Indicator */}
      {isAnalyzing && (
        <div className="bg-white p-8 rounded-lg border border-slate-200 shadow-xs">
          <LoadingSpinner label="Extracting topic embeddings and retrieving personalized cognitive context..." />
        </div>
      )}

      {/* Analysis Results Display */}
      {!isAnalyzing && analysisResult && (
        <div className="space-y-6">
          {/* Detected Topic Card */}
          <SectionCard
            title="Detected Topic & Canonical Ontology"
            badge={
              topicAbstained ? (
                <StatusBadge label="Topic Abstention" variant="warning" size="sm" dot />
              ) : (
                <StatusBadge label="Topic Resolved" variant="success" size="sm" dot />
              )
            }
          >
            {topicAbstained ? (
              <div className="p-4 rounded-md bg-amber-50/70 border border-amber-200 text-amber-900 text-xs space-y-2">
                <div className="flex items-center gap-2 font-semibold">
                  <svg className="w-4 h-4 text-amber-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>Topic could not be identified confidently. Please provide a more specific mathematics question.</span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pt-2 border-t border-amber-200 text-[11px] text-amber-800 font-mono">
                  <div>Confidence: {(analysisResult.topic.confidence * 100).toFixed(1)}%</div>
                  <div>Method: {analysisResult.topic.method}</div>
                  <div>Needs Review: {analysisResult.topic.needs_review ? 'Yes' : 'No'}</div>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-3.5 bg-slate-50 rounded-md border border-slate-200">
                  <div>
                    <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block">
                      Canonical Skill
                    </span>
                    <span className="text-sm font-bold text-slate-900">
                      {analysisResult.topic.display_name}
                    </span>
                    <span className="text-xs text-slate-500 font-mono block mt-0.5">
                      {analysisResult.topic.canonical_skill_name}
                    </span>
                  </div>
                  <div className="text-right sm:text-right shrink-0">
                    <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block">
                      Confidence Score
                    </span>
                    <span className="text-sm font-bold text-emerald-600 font-mono">
                      {(analysisResult.topic.confidence * 100).toFixed(1)}%
                    </span>
                    <span className="text-[11px] text-slate-400 block">
                      Method: {analysisResult.topic.method}
                    </span>
                  </div>
                </div>

                {analysisResult.topic.top_candidates && analysisResult.topic.top_candidates.length > 0 && (
                  <div>
                    <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-2">
                      Top Canonical Centroid Candidates
                    </span>
                    <div className="space-y-1.5">
                      {analysisResult.topic.top_candidates.slice(0, 3).map((candidate, idx) => (
                        <div
                          key={candidate.skill_id || idx}
                          className="flex items-center justify-between px-3 py-1.5 bg-white rounded border border-slate-200 text-xs font-mono"
                        >
                          <span className="truncate text-slate-700">
                            {candidate.display_name}
                          </span>
                          <span className="text-slate-500 shrink-0 ml-2">
                            {(candidate.score * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </SectionCard>

          {/* Dynamic Context & Learning State Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Learning State Card */}
            <SectionCard
              title="Learning State"
              badge={
                <StatusBadge
                  label={analysisResult.learning_state?.learning_state || 'UNKNOWN'}
                  variant={getLearningStateVariant(analysisResult.learning_state?.learning_state)}
                  size="sm"
                  dot
                />
              }
            >
              {analysisResult.learning_state ? (
                <div className="space-y-3 text-xs">
                  <div className="grid grid-cols-2 gap-2 p-3 bg-slate-50 rounded border border-slate-200">
                    <div>
                      <span className="text-slate-400 block text-[10px] uppercase font-semibold">
                        Evidence Level
                      </span>
                      <span className="font-semibold text-slate-800">
                        {analysisResult.learning_state.evidence_level}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[10px] uppercase font-semibold">
                        Evidence Strength
                      </span>
                      <span className="font-semibold text-slate-800 font-mono">
                        {String(analysisResult.learning_state.evidence_strength)}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[10px] uppercase font-semibold">
                        Behavioral Coverage
                      </span>
                      <span className="font-semibold text-slate-800 font-mono">
                        {String(analysisResult.learning_state.behavioural_coverage)}
                      </span>
                    </div>
                    <div>
                      <span className="text-slate-400 block text-[10px] uppercase font-semibold">
                        Recent Interactions
                      </span>
                      <span className="font-semibold text-slate-800 font-mono">
                        {analysisResult.learning_state.recent_interaction_count}
                      </span>
                    </div>
                  </div>

                  <div className="text-[11px] text-slate-500 font-mono space-y-0.5">
                    <div>Attempt Observations: {analysisResult.learning_state.attempt_observation_count}</div>
                    <div>Hint Observations: {analysisResult.learning_state.hint_observation_count}</div>
                    <div>Response Time Observations: {analysisResult.learning_state.response_time_observation_count}</div>
                  </div>
                </div>
              ) : (
                <EmptyState
                  title="No Dynamic State Found"
                  message="No prior learning-state snapshot recorded for this student on the resolved skill."
                  compact
                />
              )}
            </SectionCard>

            {/* Memory Context Card */}
            <SectionCard
              title="Memory Context Projections"
              badge={<StatusBadge label="Multi-Tier Memory" variant="neutral" size="sm" />}
            >
              <div className="space-y-3 text-xs">
                <div className="p-2.5 bg-slate-50 rounded border border-slate-200">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
                    Short-Term Memory (STM)
                  </span>
                  {analysisResult.short_term_memory ? (
                    <div className="font-mono text-[11px] text-slate-700 space-y-0.5">
                      <div>Recent Interactions: {analysisResult.short_term_memory.recent_interaction_count ?? 0}</div>
                      <div>Recent Skills: {analysisResult.short_term_memory.recent_skill_ids?.length ?? 0}</div>
                    </div>
                  ) : (
                    <span className="text-slate-400 italic text-[11px]">No active short-term memory buffer.</span>
                  )}
                </div>

                <div className="p-2.5 bg-slate-50 rounded border border-slate-200">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
                    Long-Term Memory (LTM)
                  </span>
                  {analysisResult.long_term_memory ? (
                    <div className="font-mono text-[11px] text-slate-700 space-y-0.5">
                      <div>Mastery Level: {analysisResult.long_term_memory.mastery_level !== undefined ? (Number(analysisResult.long_term_memory.mastery_level) * 100).toFixed(1) + '%' : 'N/A'}</div>
                      <div>Retention Strength: {analysisResult.long_term_memory.retention_strength ?? 'N/A'}</div>
                    </div>
                  ) : (
                    <span className="text-slate-400 italic text-[11px]">No long-term memory projection.</span>
                  )}
                </div>

                <div className="p-2.5 bg-slate-50 rounded border border-slate-200">
                  <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400 block mb-1">
                    Concept Memory
                  </span>
                  {analysisResult.concept_memory ? (
                    <div className="font-mono text-[11px] text-slate-700 space-y-0.5">
                      <div>Interactions: {analysisResult.concept_memory.interaction_count ?? 0}</div>
                      <div>Successes: {analysisResult.concept_memory.successful_interactions ?? 0}</div>
                    </div>
                  ) : (
                    <span className="text-slate-400 italic text-[11px]">No concept memory record.</span>
                  )}
                </div>
              </div>
            </SectionCard>
          </div>
        </div>
      )}

      {/* SECTION 2: ASSESSMENT EVALUATION & MEMORY UPDATE */}
      <SectionCard
        title="2. Assessment Evaluation & Live Memory Update"
        subtitle="Submit evaluated assessment questions to update the student's learning state, STM, LTM, and Concept Memory in PostgreSQL."
        badge={<StatusBadge label="Assessment Engine" variant="indigo" size="sm" />}
      >
        <form onSubmit={handleSubmitAssessment} className="space-y-5">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label
                htmlFor="assessment-topic"
                className="block text-xs font-semibold text-slate-700 mb-1"
              >
                Assessment Topic <span className="text-rose-500">*</span>
              </label>
              <input
                id="assessment-topic"
                type="text"
                value={assessmentTopic}
                onChange={(e) => setAssessmentTopic(e.target.value)}
                placeholder="e.g. Algebra"
                className="w-full px-3 py-1.5 text-xs rounded-md border border-slate-300 bg-white shadow-xs focus:outline-none focus:ring-2 focus:ring-slate-900"
                required
              />
            </div>
            <div>
              <label
                htmlFor="assessment-subtopic"
                className="block text-xs font-semibold text-slate-700 mb-1"
              >
                Subtopic <span className="text-slate-400 font-normal">(Optional)</span>
              </label>
              <input
                id="assessment-subtopic"
                type="text"
                value={assessmentSubtopic}
                onChange={(e) => setAssessmentSubtopic(e.target.value)}
                placeholder="e.g. Linear equations"
                className="w-full px-3 py-1.5 text-xs rounded-md border border-slate-300 bg-white shadow-xs focus:outline-none focus:ring-2 focus:ring-slate-900"
              />
            </div>
          </div>

          {/* Assessment Questions List */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                Assessment Questions ({questions.length})
              </span>
              <button
                type="button"
                onClick={handleAddQuestion}
                className="inline-flex items-center gap-1 px-2.5 py-1 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 rounded border border-slate-300 transition-colors"
              >
                + Add Question
              </button>
            </div>

            {questions.map((q, idx) => (
              <div
                key={q.id}
                className="p-4 bg-slate-50/70 rounded-lg border border-slate-200 space-y-3 relative"
              >
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-slate-800">
                    Question #{idx + 1}
                  </span>
                  {questions.length > 1 && (
                    <button
                      type="button"
                      onClick={() => handleRemoveQuestion(q.id)}
                      className="text-xs text-rose-600 hover:text-rose-800 font-medium"
                    >
                      Remove
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="sm:col-span-3">
                    <label className="block text-[11px] font-semibold text-slate-600 mb-0.5">
                      Question Text
                    </label>
                    <input
                      type="text"
                      value={q.question}
                      onChange={(e) => handleQuestionChange(q.id, 'question', e.target.value)}
                      placeholder="e.g. Solve for x: 3x = 15"
                      className="w-full px-2.5 py-1 text-xs rounded border border-slate-300 bg-white focus:outline-none focus:ring-1 focus:ring-slate-900"
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-[11px] font-semibold text-slate-600 mb-0.5">
                      Student Answer
                    </label>
                    <input
                      type="text"
                      value={q.student_answer}
                      onChange={(e) => handleQuestionChange(q.id, 'student_answer', e.target.value)}
                      placeholder="e.g. 5"
                      className="w-full px-2.5 py-1 text-xs rounded border border-slate-300 bg-white focus:outline-none focus:ring-1 focus:ring-slate-900"
                    />
                  </div>

                  <div>
                    <label className="block text-[11px] font-semibold text-slate-600 mb-0.5">
                      Expected Answer
                    </label>
                    <input
                      type="text"
                      value={q.expected_answer}
                      onChange={(e) => handleQuestionChange(q.id, 'expected_answer', e.target.value)}
                      placeholder="e.g. 5"
                      className="w-full px-2.5 py-1 text-xs rounded border border-slate-300 bg-white focus:outline-none focus:ring-1 focus:ring-slate-900"
                    />
                  </div>

                  <div>
                    <label className="block text-[11px] font-semibold text-slate-600 mb-0.5">
                      Evaluation Result
                    </label>
                    <select
                      value={q.is_correct ? 'true' : 'false'}
                      onChange={(e) => handleQuestionChange(q.id, 'is_correct', e.target.value === 'true')}
                      className={`w-full px-2.5 py-1 text-xs rounded border font-semibold focus:outline-none focus:ring-1 focus:ring-slate-900 ${
                        q.is_correct ? 'bg-emerald-50 text-emerald-800 border-emerald-300' : 'bg-rose-50 text-rose-800 border-rose-300'
                      }`}
                    >
                      <option value="true">Correct (Yes)</option>
                      <option value="false">Incorrect (No)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-[11px] font-semibold text-slate-600 mb-0.5">
                      Identified Error <span className="text-slate-400 font-normal">(Optional)</span>
                    </label>
                    <input
                      type="text"
                      value={q.identified_error}
                      onChange={(e) => handleQuestionChange(q.id, 'identified_error', e.target.value)}
                      placeholder="e.g. Subtracted instead of dividing"
                      className="w-full px-2.5 py-1 text-xs rounded border border-slate-300 bg-white focus:outline-none focus:ring-1 focus:ring-slate-900"
                    />
                  </div>

                  <div>
                    <label className="block text-[11px] font-semibold text-slate-600 mb-0.5">
                      Attempts / Hints
                    </label>
                    <div className="flex gap-2">
                      <input
                        type="number"
                        min="0"
                        value={q.attempt_count}
                        onChange={(e) => handleQuestionChange(q.id, 'attempt_count', e.target.value)}
                        placeholder="Att (1)"
                        className="w-1/2 px-2 py-1 text-xs rounded border border-slate-300 bg-white focus:outline-none focus:ring-1 focus:ring-slate-900"
                      />
                      <input
                        type="number"
                        min="0"
                        value={q.hint_count}
                        onChange={(e) => handleQuestionChange(q.id, 'hint_count', e.target.value)}
                        placeholder="Hints (0)"
                        className="w-1/2 px-2 py-1 text-xs rounded border border-slate-300 bg-white focus:outline-none focus:ring-1 focus:ring-slate-900"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-[11px] font-semibold text-slate-600 mb-0.5">
                      Response Time (ms)
                    </label>
                    <input
                      type="number"
                      min="0"
                      value={q.response_time_ms}
                      onChange={(e) => handleQuestionChange(q.id, 'response_time_ms', e.target.value)}
                      placeholder="e.g. 4200"
                      className="w-full px-2.5 py-1 text-xs rounded border border-slate-300 bg-white focus:outline-none focus:ring-1 focus:ring-slate-900"
                    />
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Submit Action Bar */}
          <div className="flex items-center justify-between pt-2 border-t border-slate-200">
            <span className="text-[11px] text-slate-400">
              Executes ML feature builder, model inference, and atomic database persistence.
            </span>
            <button
              type="submit"
              disabled={isSubmitting}
              className="inline-flex items-center justify-center px-5 py-2 text-xs font-semibold text-white bg-slate-900 hover:bg-slate-800 disabled:bg-slate-400 rounded-md shadow-xs transition-colors focus:outline-none focus:ring-2 focus:ring-slate-900"
            >
              {isSubmitting ? 'Updating Memory...' : 'Submit Assessment'}
            </button>
          </div>
        </form>
      </SectionCard>

      {/* Submission Error Banner */}
      {submitError && (
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-start gap-2.5">
          <svg className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div>
            <span className="font-semibold block">Memory Update Failed</span>
            <span>{submitError}</span>
          </div>
        </div>
      )}

      {/* Updated Memory Notification Card */}
      {updateResult && (
        <div className="p-5 bg-emerald-50/80 border border-emerald-200 rounded-lg space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" />
              <span className="text-sm font-bold text-emerald-950">
                Cognitive Memory Updated Successfully
              </span>
            </div>
            <StatusBadge
              label={`State: ${updateResult.learning_state}`}
              variant={getLearningStateVariant(updateResult.learning_state)}
              size="sm"
              dot
            />
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono pt-1 text-slate-800">
            <div className="p-2.5 bg-white rounded border border-emerald-200">
              <span className="text-slate-500 block text-[10px] uppercase font-semibold">
                Evidence Strength
              </span>
              <span className="font-bold text-slate-900">{updateResult.evidence_strength}</span>
            </div>
            <div className="p-2.5 bg-white rounded border border-emerald-200">
              <span className="text-slate-500 block text-[10px] uppercase font-semibold">
                Coverage
              </span>
              <span className="font-bold text-slate-900 truncate block">
                {updateResult.behavioural_coverage.replace('_COVERAGE', '')}
              </span>
            </div>
            <div className="p-2.5 bg-white rounded border border-emerald-200">
              <span className="text-slate-500 block text-[10px] uppercase font-semibold">
                Recent Interactions
              </span>
              <span className="font-bold text-slate-900">{updateResult.recent_interaction_count}</span>
            </div>
            <div className="p-2.5 bg-white rounded border border-emerald-200">
              <span className="text-slate-500 block text-[10px] uppercase font-semibold">
                Misconceptions
              </span>
              <span className="font-bold text-slate-900">{updateResult.misconception_count}</span>
            </div>
          </div>

          <div className="text-[11px] text-emerald-800 font-mono flex items-center justify-between pt-1">
            <span>Assessment ID: #{updateResult.assessment_id} • Snapshot ID: #{updateResult.snapshot_id}</span>
            <span>Persisted to PostgreSQL schema 'student_memory'</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default StudentPage
