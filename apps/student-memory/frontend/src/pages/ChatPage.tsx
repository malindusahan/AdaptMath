import React, { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { getStoredAuthUser, logout } from '../api/authApi'
import {
  createSession,
  getQuestionContext,
  listSessions,
  listSkills,
  submitMemoryUpdate,
  updateSession,
} from '../api/memoryApi'
import {
  AssessmentMemoryUpdateRequest,
  QuestionContextResponse,
  SessionItemResponse,
  SkillItem,
} from '../types/api'

interface ChatMessage {
  id: string
  sender: 'student' | 'assistant'
  text: string
  timestamp: string
  context?: QuestionContextResponse
  isAbstention?: boolean
}

export const ChatPage: React.FC = () => {
  const navigate = useNavigate()
  const authUser = getStoredAuthUser()

  const [sessions, setSessions] = useState<SessionItemResponse[]>([])
  const [activeSession, setActiveSession] = useState<SessionItemResponse | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [questionInput, setQuestionInput] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [isAssessmentOpen, setIsAssessmentOpen] = useState(false)

  // Assessment demo state
  const [availableSkills, setAvailableSkills] = useState<SkillItem[]>([])
  const [isTopicDropdownOpen, setIsTopicDropdownOpen] = useState(false)
  const [topicSearchQuery, setTopicSearchQuery] = useState('')
  const [assessmentTopic, setAssessmentTopic] = useState('math :: algebra :: linear equations')
  const [studentAnswer, setStudentAnswer] = useState('x = 6')
  const [isCorrect, setIsCorrect] = useState(true)
  const [errorDescription, setErrorDescription] = useState('')
  const [isUpdatingMemory, setIsUpdatingMemory] = useState(false)
  const [updateSuccessNote, setUpdateSuccessNote] = useState<string | null>(null)

  const getActiveSessionStorageKey = () => {
    return `last_active_session_${authUser?.username || 'user'}`
  }

  const getSessionStorageKey = (sessId: string) => {
    return `chat_messages_${authUser?.username || 'user'}_${sessId}`
  }

  const switchSession = (session: SessionItemResponse) => {
    setActiveSession(session)
    localStorage.setItem(getActiveSessionStorageKey(), session.session_id)
    const stored = localStorage.getItem(getSessionStorageKey(session.session_id))
    if (stored) {
      try {
        setMessages(JSON.parse(stored))
      } catch {
        setMessages([])
      }
    } else {
      setMessages([])
    }
  }

  useEffect(() => {
    if (!authUser) {
      navigate('/login')
      return
    }
    loadSessions()
    loadSkills()
  }, [])

  // Auto-save messages to local storage whenever messages update for active session
  useEffect(() => {
    if (activeSession && messages.length > 0) {
      localStorage.setItem(getSessionStorageKey(activeSession.session_id), JSON.stringify(messages))
    }
  }, [messages, activeSession])

  const loadSkills = async () => {
    try {
      const skills = await listSkills()
      setAvailableSkills(skills)
    } catch (err) {
      console.error('Failed to load skills list', err)
    }
  }

  const loadSessions = async () => {
    try {
      const res = await listSessions(30)
      setSessions(res)
      if (res.length > 0) {
        const lastActiveId = localStorage.getItem(getActiveSessionStorageKey())
        const matched = res.find((s) => s.session_id === lastActiveId)
        if (matched) {
          switchSession(matched)
        } else {
          switchSession(res[0])
        }
      } else {
        handleNewChat()
      }
    } catch (err: any) {
      if (err?.response?.status === 401) {
        await logout()
        navigate('/login')
        return
      }
      handleNewChat()
    }
  }

  const handleNewChat = async () => {
    // If the active session is already fresh with 0 messages, just stay on it
    if (activeSession && messages.length === 0 && (activeSession.external_session_id.startsWith('chat_') || activeSession.external_session_id === 'New Chat')) {
      return
    }

    try {
      const newSess = await createSession()
      setSessions((prev) => [newSess, ...prev])
      switchSession(newSess)
    } catch (err: any) {
      console.error('Failed to create session', err)
      if (err?.response?.status === 401) {
        await logout()
        navigate('/login')
      }
    }
  }

  const handleSendQuestion = async (textToSend?: string) => {
    const q = (textToSend || questionInput).trim()
    if (!q || isSending) return

    let currentSess = activeSession
    if (!currentSess) {
      try {
        currentSess = await createSession()
        setSessions((prev) => [currentSess!, ...prev])
        switchSession(currentSess)
      } catch (err: any) {
        if (err?.response?.status === 401) {
          await logout()
          navigate('/login')
          return
        }
      }
    }

    const studentMsg: ChatMessage = {
      id: `msg_${Date.now()}`,
      sender: 'student',
      text: q,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }

    const updatedMessages = [...messages, studentMsg]
    setMessages(updatedMessages)
    setQuestionInput('')
    setIsSending(true)

    // Immediately move the active chat to the top of the sidebar list
    if (currentSess) {
      setSessions((prev) => [
        currentSess!,
        ...prev.filter((s) => s.session_id !== currentSess!.session_id),
      ])
    }

    try {
      const contextRes = await getQuestionContext({
        student_id: authUser?.student_id || authUser?.username || 'student',
        session_id: currentSess?.session_id || 'default',
        question: q,
      })

      const isAbstention = contextRes.topic.needs_review || !contextRes.topic.canonical_skill_name
      let assistantText = ''
      const topicName = contextRes.topic.display_name || contextRes.topic.canonical_skill_name || 'Mathematics'

      if (isAbstention) {
        assistantText = `Hello! 👋 I'm your Math Learning Assistant. I can help you understand and solve math problems across topics like Algebra, Linear Equations, Pythagorean Theorem, Fractions, Geometry, Calculus, and more! What math question or concept would you like to work on today?`
      } else {
        const stateName = contextRes.learning_state?.learning_state
        if (stateName) {
          assistantText = `I identified your question under "${topicName}". Your personalized learning state is ${stateName} with relevant history loaded for your session!`
        } else {
          assistantText = `I identified your question under "${topicName}". Let's start working through this math concept together!`
        }

        // Dynamically rename chat session to the extracted topic name if not already renamed
        if (
          currentSess &&
          (currentSess.external_session_id.startsWith('chat_') ||
            currentSess.external_session_id === 'New Chat')
        ) {
          try {
            await updateSession(currentSess.session_id, topicName)
            const renamedSess = { ...currentSess, external_session_id: topicName }
            setActiveSession(renamedSess)
            setSessions((prev) => [
              renamedSess,
              ...prev.filter((s) => s.session_id !== renamedSess.session_id),
            ])
          } catch (err) {
            console.error('Failed to update session title', err)
          }
        }
      }

      const assistantMsg: ChatMessage = {
        id: `msg_resp_${Date.now()}`,
        sender: 'assistant',
        text: assistantText,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        context: contextRes,
        isAbstention,
      }

      setMessages((prev) => [...prev, assistantMsg])
    } catch (err: any) {
      if (err?.response?.status === 401) {
        await logout()
        navigate('/login')
        return
      }
      const errorMsg: ChatMessage = {
        id: `msg_err_${Date.now()}`,
        sender: 'assistant',
        text: `Error analyzing question: ${err.response?.data?.detail || 'Unable to reach memory context service.'}`,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }
      setMessages((prev) => [...prev, errorMsg])
    } finally {
      setIsSending(false)
    }
  }

  const handleLogout = async () => {
    await logout()
    navigate('/login')
  }

  const handleAssessmentSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsUpdatingMemory(true)
    setUpdateSuccessNote(null)

    try {
      const payload: AssessmentMemoryUpdateRequest = {
        student_id: authUser?.student_id || authUser?.username || 'demo_developing',
        topic: assessmentTopic,
        assessment_questions: [
          {
            question_id: `q_${Date.now()}`,
            question: 'Solve problem for this topic',
            student_answer: studentAnswer,
            expected_answer: 'x = 6',
            is_correct: isCorrect,
            identified_error: isCorrect ? null : errorDescription || 'Calculation error',
            attempt_count: isCorrect ? 1 : 2,
            hint_count: isCorrect ? 0 : 1,
            response_time_ms: 4500.0,
          },
        ],
      }

      const res = await submitMemoryUpdate(payload)
      setUpdateSuccessNote(
        `Memory updated! New Learning State: ${res.learning_state || 'UPDATED'}`
      )
    } catch (err: any) {
      setUpdateSuccessNote(`Update error: ${err.response?.data?.detail || err.message || 'Failed to update memory.'}`)
    } finally {
      setIsUpdatingMemory(false)
    }
  }

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* ----------------- Left Sidebar ----------------- */}
      <div className="w-72 bg-slate-900 border-r border-slate-800 flex flex-col flex-shrink-0">
        {/* Header */}
        <div className="p-4 border-b border-slate-800">
          <button
            onClick={handleNewChat}
            id="btn-new-chat"
            className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2.5 px-4 rounded-xl transition-all shadow-md shadow-indigo-600/20 flex items-center justify-center gap-2 text-sm"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>+ New Chat</span>
          </button>
        </div>

        {/* Sessions List */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1.5 custom-scrollbar">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 px-3 py-1">
            Recent Chats
          </div>
          {sessions.map((sess) => {
            const isSelected = activeSession?.session_id === sess.session_id
            return (
              <button
                key={sess.session_id}
                onClick={() => switchSession(sess)}
                className={`w-full text-left px-3 py-2.5 rounded-xl text-xs transition-all flex items-center gap-2.5 ${
                  isSelected
                    ? 'bg-slate-800 text-indigo-300 font-medium border border-slate-700'
                    : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200 border border-transparent'
                }`}
              >
                <svg className="w-4 h-4 flex-shrink-0 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
                <span className="truncate">{sess.external_session_id}</span>
              </button>
            )
          })}
        </div>

        {/* User Profile Footer */}
        <div className="p-3.5 border-t border-slate-800 bg-slate-900/60 flex items-center justify-between">
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-8 h-8 rounded-lg bg-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold text-xs">
              {authUser?.username?.[0]?.toUpperCase() || 'S'}
            </div>
            <div className="min-w-0">
              <div className="text-xs font-semibold text-white truncate">{authUser?.username}</div>
              <div className="text-[10px] text-slate-400 truncate">
                {authUser?.age !== null ? `Age: ${authUser?.age}` : 'Student'}
              </div>
            </div>
          </div>
          <button
            onClick={handleLogout}
            id="btn-logout"
            title="Log Out"
            className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-rose-400 transition-all"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
            </svg>
          </button>
        </div>
      </div>

      {/* ----------------- Main Chat Area ----------------- */}
      <div className="flex-1 flex flex-col min-w-0 bg-slate-950">
        {/* Top bar */}
        <div className="h-14 border-b border-slate-800 px-6 flex items-center justify-between bg-slate-900/50 backdrop-blur-md">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-white">Mathematics Personalization Assistant</h2>
            {activeSession && (
              <span className="text-[11px] font-mono bg-slate-800 text-slate-400 px-2 py-0.5 rounded-md">
                {activeSession.external_session_id}
              </span>
            )}
          </div>

          <button
            onClick={() => setIsAssessmentOpen(!isAssessmentOpen)}
            className="text-xs px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-all flex items-center gap-1.5"
          >
            <svg className="w-3.5 h-3.5 text-indigo-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            <span>{isAssessmentOpen ? 'Hide Assessment Panel' : 'Simulate Assessment Update'}</span>
          </button>
        </div>

        {/* Chat message feed */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center max-w-lg mx-auto text-center space-y-4 text-slate-400">
              <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center">
                <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-white">Ask Any Mathematics Question</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                The Topic Extractor will automatically map your question to canonical skills and retrieve your persistent Short-Term, Long-Term, and Concept Memory.
              </p>
              <div className="grid grid-cols-2 gap-2 text-left w-full pt-2">
                {[
                  'How do I solve 3x + 5 = 20?',
                  'What is the hypotenuse if legs are 3 and 4?',
                  'Can you explain slope formula m = (y2-y1)/(x2-x1)?',
                  'Can you help me?',
                ].map((preset) => (
                  <button
                    key={preset}
                    onClick={() => handleSendQuestion(preset)}
                    className="p-3 bg-slate-900 border border-slate-800 hover:border-indigo-500/40 rounded-xl text-xs text-slate-300 hover:text-white transition-all text-left"
                  >
                    "{preset}"
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex gap-3 max-w-2xl ${
                  msg.sender === 'student' ? 'ml-auto flex-row-reverse' : ''
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-xl flex-shrink-0 flex items-center justify-center text-xs font-bold ${
                    msg.sender === 'student'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-slate-800 text-emerald-400 border border-slate-700'
                  }`}
                >
                  {msg.sender === 'student' ? authUser?.username?.[0]?.toUpperCase() || 'S' : 'AI'}
                </div>

                <div
                  className={`space-y-3 rounded-2xl p-4 text-xs ${
                    msg.sender === 'student'
                      ? 'bg-indigo-600 text-white'
                      : 'bg-slate-900 border border-slate-800 text-slate-200'
                  }`}
                >
                  <p className="leading-relaxed whitespace-pre-wrap">{msg.text}</p>

                  {/* Context Cards for Assistant */}
                  {msg.context && !msg.isAbstention && (
                    <div className="mt-3 pt-3 border-t border-slate-800/80 space-y-2.5">
                      {/* Topic Identification */}
                      <div className="flex items-center justify-between bg-slate-950/60 p-2.5 rounded-lg border border-slate-800 text-[11px]">
                        <span className="text-slate-400">Identified Skill:</span>
                        <span className="font-semibold text-indigo-300">
                          {msg.context.topic.display_name || msg.context.topic.canonical_skill_name}
                        </span>
                      </div>

                      {/* Learning State & Memory Status */}
                      <div className="grid grid-cols-2 gap-2 text-[11px]">
                        <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
                          <div className="text-slate-500 mb-1">Learning State</div>
                          <div
                            className={`font-semibold ${
                              msg.context.learning_state?.learning_state === 'STRONG'
                                ? 'text-emerald-400'
                                : msg.context.learning_state?.learning_state === 'DEVELOPING'
                                ? 'text-amber-400'
                                : msg.context.learning_state?.learning_state === 'NEEDS_SUPPORT'
                                ? 'text-rose-400'
                                : 'text-slate-400'
                            }`}
                          >
                            {msg.context.learning_state?.learning_state || 'COLD START (No Prior History)'}
                          </div>
                        </div>

                        <div className="bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
                          <div className="text-slate-500 mb-1">Concept Accuracy</div>
                          <div className="font-semibold text-white">
                            {typeof msg.context.concept_memory?.accuracy === 'number'
                              ? `${Math.round(Number(msg.context.concept_memory.accuracy) * 100)}%`
                              : 'No prior data'}
                          </div>
                        </div>
                      </div>

                      {/* Learning Challenges / Misconceptions (Shown only when student needs support or is developing) */}
                      {msg.context.learning_state?.learning_state !== 'STRONG' &&
                        msg.context.misconceptions &&
                        msg.context.misconceptions.length > 0 && (
                          <div className="bg-amber-500/10 border border-amber-500/20 p-2 rounded-lg text-[10px] text-amber-300 flex items-center gap-1.5">
                            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shrink-0" />
                            <span>
                              <strong className="font-medium text-amber-200">Personalized Focus Area: </strong>
                              {msg.context.misconceptions[0].display_error}
                            </span>
                          </div>
                        )}
                    </div>
                  )}

                  <div className="text-[10px] text-slate-500 text-right">{msg.timestamp}</div>
                </div>
              </div>
            ))
          )}

          {isSending && (
            <div className="flex gap-3 max-w-md">
              <div className="w-8 h-8 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center text-xs text-emerald-400 font-bold">
                AI
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-2xl p-4 text-xs text-slate-400 flex items-center gap-2">
                <div className="w-3.5 h-3.5 border-2 border-slate-600 border-t-indigo-400 rounded-full animate-spin" />
                <span>Extracting topic & retrieving memory context...</span>
              </div>
            </div>
          )}
        </div>

        {/* ----------------- Assessment Demo Drawer ----------------- */}
        {isAssessmentOpen && (
          <div className="border-t border-slate-800 bg-slate-900/90 p-4 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-semibold text-indigo-400 uppercase tracking-wider">
                Evaluator Assessment Simulation (Updates Real PostgreSQL Memory)
              </span>
              <button
                onClick={() => setIsAssessmentOpen(false)}
                className="text-slate-400 hover:text-white text-xs"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleAssessmentSubmit} className="grid grid-cols-3 gap-3 text-xs">
              <div className="relative">
                <div className="flex items-center justify-between mb-1">
                  <label className="block text-slate-400">Topic</label>
                  <span className="text-[10px] text-indigo-400 font-mono">
                    {availableSkills.length > 0 ? `${availableSkills.length} Available` : '111 Topics'}
                  </span>
                </div>
                
                <button
                  type="button"
                  onClick={() => setIsTopicDropdownOpen((prev) => !prev)}
                  className="w-full bg-slate-950 border border-slate-800 hover:border-slate-700 rounded-lg px-2.5 py-1.5 text-white flex items-center justify-between text-left focus:outline-none focus:border-indigo-500 transition-all"
                >
                  <div className="truncate pr-2">
                    <span className="font-medium text-white block truncate">
                      {availableSkills.find((s) => s.canonical_name === assessmentTopic || s.display_name === assessmentTopic)?.display_name || assessmentTopic}
                    </span>
                    <span className="text-[10px] text-slate-500 block truncate">
                      {assessmentTopic}
                    </span>
                  </div>
                  <span className="text-slate-400 text-xs shrink-0">▼</span>
                </button>

                {/* Searchable Dropdown Popover */}
                {isTopicDropdownOpen && (
                  <>
                    <div
                      className="fixed inset-0 z-20"
                      onClick={() => setIsTopicDropdownOpen(false)}
                    />
                    <div className="absolute left-0 bottom-full mb-1 w-80 md:w-96 bg-slate-900 border border-slate-700 rounded-xl shadow-2xl z-30 p-2 text-xs flex flex-col max-h-72">
                      {/* Search Bar */}
                      <div className="relative mb-2">
                        <input
                          type="text"
                          value={topicSearchQuery}
                          onChange={(e) => setTopicSearchQuery(e.target.value)}
                          placeholder="Search 111 topics (e.g. slope, linear, fraction)..."
                          className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-3 pr-7 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500"
                          autoFocus
                        />
                        {topicSearchQuery && (
                          <button
                            type="button"
                            onClick={() => setTopicSearchQuery('')}
                            className="absolute right-2 top-1.5 text-slate-500 hover:text-white text-xs"
                          >
                            ✕
                          </button>
                        )}
                      </div>

                      {/* Filtered Skills List */}
                      <div className="overflow-y-auto space-y-1 flex-1 pr-1 custom-scrollbar">
                        {availableSkills
                          .filter((skill) => {
                            const query = topicSearchQuery.toLowerCase().trim()
                            if (!query) return true
                            return (
                              skill.display_name.toLowerCase().includes(query) ||
                              skill.canonical_name.toLowerCase().includes(query) ||
                              (skill.category && skill.category.toLowerCase().includes(query)) ||
                              skill.skill_code.toLowerCase().includes(query)
                            )
                          })
                          .map((skill) => {
                            const isSelected = assessmentTopic === skill.canonical_name
                            return (
                              <button
                                key={skill.skill_id}
                                type="button"
                                onClick={() => {
                                  setAssessmentTopic(skill.canonical_name)
                                  setIsTopicDropdownOpen(false)
                                  setTopicSearchQuery('')
                                }}
                                className={`w-full text-left p-2 rounded-lg transition-all flex items-center justify-between ${
                                  isSelected
                                    ? 'bg-indigo-600/30 border border-indigo-500/50 text-white'
                                    : 'hover:bg-slate-800/80 text-slate-300 hover:text-white'
                                }`}
                              >
                                <div className="truncate pr-2">
                                  <div className="font-semibold text-xs text-white truncate">
                                    {skill.display_name}
                                  </div>
                                  <div className="text-[10px] text-slate-400 truncate">
                                    {skill.category ? `${skill.category} • ` : ''}{skill.skill_code}
                                  </div>
                                </div>
                                {isSelected && (
                                  <span className="text-emerald-400 text-xs font-bold">✓</span>
                                )}
                              </button>
                            )
                          })}
                        {availableSkills.filter((skill) => {
                          const query = topicSearchQuery.toLowerCase().trim()
                          if (!query) return true
                          return (
                            skill.display_name.toLowerCase().includes(query) ||
                            skill.canonical_name.toLowerCase().includes(query) ||
                            (skill.category && skill.category.toLowerCase().includes(query)) ||
                            skill.skill_code.toLowerCase().includes(query)
                          )
                        }).length === 0 && (
                          <div className="text-center py-4 text-slate-500 text-xs">
                            No matching topics found for "{topicSearchQuery}"
                          </div>
                        )}
                      </div>
                    </div>
                  </>
                )}
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Student Answer</label>
                <input
                  type="text"
                  value={studentAnswer}
                  onChange={(e) => setStudentAnswer(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-white"
                />
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Correctness</label>
                <div className="flex gap-2 pt-1">
                  <button
                    type="button"
                    onClick={() => setIsCorrect(true)}
                    className={`px-3 py-1 rounded-md text-xs font-semibold ${
                      isCorrect ? 'bg-emerald-600 text-white' : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    CORRECT
                  </button>
                  <button
                    type="button"
                    onClick={() => setIsCorrect(false)}
                    className={`px-3 py-1 rounded-md text-xs font-semibold ${
                      !isCorrect ? 'bg-rose-600 text-white' : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    INCORRECT
                  </button>
                </div>
              </div>

              {!isCorrect && (
                <div className="col-span-3">
                  <label className="block text-slate-400 mb-1">Identified Misconception / Error</label>
                  <input
                    type="text"
                    value={errorDescription}
                    onChange={(e) => setErrorDescription(e.target.value)}
                    placeholder="e.g. Subtracted instead of divided"
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-2.5 py-1.5 text-white"
                  />
                </div>
              )}

              <div className="col-span-3 flex items-center justify-between pt-1">
                {updateSuccessNote && (
                  <span className="text-xs text-emerald-400">{updateSuccessNote}</span>
                )}
                <button
                  type="submit"
                  disabled={isUpdatingMemory}
                  className="ml-auto bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-1.5 rounded-lg text-xs font-medium"
                >
                  {isUpdatingMemory ? 'Saving to Postgres...' : 'Submit Evaluator Assessment'}
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Input box */}
        <div className="p-4 border-t border-slate-800 bg-slate-900/50">
          <form
            onSubmit={(e) => {
              e.preventDefault()
              handleSendQuestion()
            }}
            className="flex gap-2"
          >
            <input
              type="text"
              id="input-student-question"
              value={questionInput}
              onChange={(e) => setQuestionInput(e.target.value)}
              placeholder="Ask a mathematics question (e.g. How do I solve 3x + 5 = 20?)..."
              className="flex-1 bg-slate-950 border border-slate-800 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-xl px-4 py-3 text-xs text-white placeholder-slate-500 outline-none"
            />
            <button
              type="submit"
              id="btn-send-question"
              disabled={isSending || !questionInput.trim()}
              className="bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white px-5 rounded-xl text-xs font-semibold transition-all shadow-md shadow-indigo-600/20"
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default ChatPage
