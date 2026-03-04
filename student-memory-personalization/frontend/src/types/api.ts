export interface HealthResponse {
  status: string
  service: string
}

export interface ReadinessResponse {
  status: string
  database: string
  migrations: string
  topic_extractor: string
  learning_state_model: string
}

export interface TopicCandidate {
  skill_id?: string
  skill_code?: string
  canonical_name?: string
  display_name: string
  score: number
}

export interface TopicContextItem {
  skill_id?: string | null
  skill_code?: string | null
  canonical_skill_name?: string | null
  display_name?: string | null
  confidence: number
  method: string
  needs_review: boolean
  top_candidates?: TopicCandidate[]
  model_version?: string
}

export interface MisconceptionItem {
  misconception_id: string
  normalized_error: string
  display_error: string
  occurrence_count: number
  last_seen_at: string
}

export interface LearningStateItem {
  learning_state: string
  evidence_level: string
  evidence_strength: string | number
  behavioural_coverage: string | number
  model_used?: boolean | string | null
  recent_interaction_count: number
  attempt_observation_count: number
  hint_observation_count: number
  response_time_observation_count: number
  updated_at: string
}

export interface ShortTermMemoryContext {
  student_id?: string
  session_id?: string
  recent_interaction_count?: number
  recent_skill_ids?: string[]
  recent_scores?: number[]
  last_activity_at?: string
  [key: string]: unknown
}

export interface LongTermMemoryContext {
  student_id?: string
  mastery_level?: number
  retention_strength?: number
  total_practice_sessions?: number
  total_problems_attempted?: number
  last_reviewed_at?: string
  [key: string]: unknown
}

export interface ConceptMemoryContext {
  student_id?: string
  skill_id?: string
  interaction_count?: number
  successful_interactions?: number
  unsuccessful_interactions?: number
  average_attempt_count?: number
  average_hint_count?: number
  average_response_time_ms?: number
  [key: string]: unknown
}

export interface QuestionContextResponse {
  student_id: string
  session_id: string
  question: string
  topic: TopicContextItem
  short_term_memory?: ShortTermMemoryContext | null
  long_term_memory?: LongTermMemoryContext | null
  concept_memory?: ConceptMemoryContext | null
  learning_state?: LearningStateItem | null
  misconceptions: MisconceptionItem[]
}

export interface QuestionContextRequest {
  student_id: string
  session_id: string
  question: string
}

export interface AssessmentQuestionInput {
  question_id: string
  question?: string
  student_answer?: string
  expected_answer?: string
  is_correct: boolean
  identified_error?: string | null
  attempt_count?: number | null
  hint_count?: number | null
  hint_total?: number | null
  response_time_ms?: number | null
}

export interface AssessmentMemoryUpdateRequest {
  student_id: string
  topic: string
  subtopic?: string | null
  assessment_questions: AssessmentQuestionInput[]
  identified_errors?: string[]
  overall_feedback?: string | null
}

export interface MemoryUpdateResponse {
  student_id: string
  topic: string
  subtopic?: string | null
  assessment_id: number
  snapshot_id: number
  learning_state: 'NEEDS_SUPPORT' | 'DEVELOPING' | 'STRONG' | 'UNAVAILABLE'
  evidence_level: 'COLD_START' | 'OVERALL_ONLY' | 'PARTIAL_SKILL' | 'FULL_SKILL'
  evidence_strength: 'NONE' | 'LOW' | 'MEDIUM' | 'HIGH'
  behavioural_coverage:
    | 'FULL_BEHAVIOURAL_COVERAGE'
    | 'PARTIAL_BEHAVIOURAL_COVERAGE'
    | 'CORRECTNESS_ONLY_COVERAGE'
  model_used: boolean
  previous_interaction_count: number
  previous_skill_interaction_count: number
  recent_interaction_count: number
  attempt_observation_count: number
  hint_observation_count: number
  response_time_observation_count: number
  misconception_count: number
  memory_updated: boolean
}

export interface BehaviouralEvidenceSummary {
  observation_count: number
  total_value: number
  average_value: number | null
}

export interface InteractionItem {
  interaction_id: string
  session_id: string
  canonical_skill_id?: string | null
  student_utterance?: string | null
  identified_error?: string | null
  is_correct: boolean
  attempt_count: number
  hint_count: number
  response_time_ms?: number | null
  created_at: string
}

export interface RepairItem {
  repair_outcome_id: string
  session_id: string
  canonical_skill_id: string
  repair_action: string
  outcome: string
  score?: number | null
  notes?: string | null
  created_at: string
}

export interface RepairOutcomeCreateRequest {
  student_id: string
  session_id: string
  skill_id: string
  interaction_id?: string | null
  repair_action: string
  outcome: 'RESOLVED' | 'PARTIALLY_RESOLVED' | 'UNRESOLVED' | string
  score?: number | null
  notes?: string | null
}

export interface RepairOutcomeResponse {
  repair_outcome_id: string
  student_id: string
  session_id: string
  canonical_skill_id: string
  interaction_id?: string | null
  repair_action: string
  outcome: string
  score?: number | null
  notes?: string | null
  created_at: string
}

export interface StudentContextResponse {
  student_id: string
  session_id?: string | null
  skill_id?: string | null
  learning_state?: Record<string, unknown> | null
  short_term_memory?: ShortTermMemoryContext | null
  long_term_memory?: LongTermMemoryContext | null
  concept_memory?: ConceptMemoryContext | null
  misconceptions: MisconceptionItem[]
  recent_interactions: InteractionItem[]
  repair_history: RepairItem[]
}

export interface TutorContextResponse {
  student_id: string
  session_id?: string | null
  skill_id?: string | null
  current_learning_state?: string | null
  evidence_strength?: string | null
  behavioural_coverage?: string | null
  recent_accuracy?: number | null
  recent_correct_count: number
  recent_incorrect_count: number
  attempt_evidence?: BehaviouralEvidenceSummary
  hint_evidence?: BehaviouralEvidenceSummary
  response_time_evidence?: BehaviouralEvidenceSummary
  misconceptions: MisconceptionItem[]
  recent_interactions: InteractionItem[]
  recent_repairs: RepairItem[]
}

export interface PlannerContextResponse {
  student_id: string
  session_id?: string | null
  skill_id?: string | null
  session_interaction_count: number
  session_accuracy?: number | null
  session_correct_count: number
  session_incorrect_count: number
  attempt_evidence?: BehaviouralEvidenceSummary
  hint_evidence?: BehaviouralEvidenceSummary
  response_time_evidence?: BehaviouralEvidenceSummary
  current_learning_state?: string | null
  evidence_strength?: string | null
  behavioural_coverage?: string | null
  concept_interaction_count: number
  concept_accuracy?: number | null
  concept_correct_count: number
  concept_incorrect_count: number
  long_term_interaction_count: number
  overall_accuracy?: number | null
  total_sessions: number
  concept_count: number
  misconceptions: MisconceptionItem[]
  recent_interactions: InteractionItem[]
}

export interface FaprContextResponse {
  student_id: string
  session_id?: string | null
  skill_id?: string | null
  current_learning_state?: string | null
  evidence_strength?: string | null
  behavioural_coverage?: string | null
  recent_accuracy?: number | null
  recent_incorrect_count: number
  attempt_evidence?: BehaviouralEvidenceSummary
  hint_evidence?: BehaviouralEvidenceSummary
  response_time_evidence?: BehaviouralEvidenceSummary
  misconceptions: MisconceptionItem[]
  recent_interactions: InteractionItem[]
  previous_repairs: RepairItem[]
  latest_student_utterance?: string | null
}

export interface SupportStrategySummary {
  repair_action: string
  observation_count: number
  successful_count: number
  partial_count: number
  failed_count: number
  success_rate: number
  average_score?: number | null
}

export interface SupportPreferenceResponse {
  student_id: string
  skill_id?: string | null
  preferred_support_style?: string | null
  status: string
  evidence_count?: number | null
  success_rate?: number | null
  strategies: SupportStrategySummary[]
}

export interface MetaSignalItem {
  signal_type: string
  student_id: string
  session_id?: string | null
  skill_id?: string | null
  interaction_id?: string | null
  timestamp: string
  confidence: number
  evidence?: Record<string, unknown>
}

export interface MetaSignalsResponse {
  student_id: string
  session_id?: string | null
  skill_id?: string | null
  total_signals: number
  signals: MetaSignalItem[]
}

export interface ApiErrorResponse {
  error_code: string
  message: string
  details?: Array<{ field?: string; message: string }>
  request_id?: string
  timestamp?: string
}

export interface SignupRequest {
  username: string
  date_of_birth: string
  password: string
  confirm_password: string
}

export interface SignupResponse {
  username: string
  student_id: string
  age: number
  role: string
}

export interface LoginRequest {
  username: string
  password: string
}

export interface LoginResponse {
  token: string
  username: string
  role: string
  student_id?: string | null
  age?: number | null
}

export interface UserMeResponse {
  user_id: string
  username: string
  role: string
  student_id?: string | null
  age?: number | null
}

export interface SessionItemResponse {
  session_id: string
  external_session_id: string
  student_id: string
  status: string
  started_at: string
}

export interface SkillItem {
  skill_id: string
  skill_code: string
  display_name: string
  canonical_name: string
  category?: string | null
  description?: string | null
}
