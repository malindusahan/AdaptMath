export type SessionStatus =
  | "student_response_required"
  | "pedagogical_move_required"
  | "assessment_required"
  | "recovery_required"
  | "complete";

export interface AssessmentQuestion {
  question_id: string;
  question: string;
}

export interface EvaluationAnswer {
  question_id?: string;
  question?: string;
  student_answer?: string;
  is_correct?: boolean;
  feedback?: string;
  identified_error?: string;
}

export interface EvaluationResult {
  correct_answers: EvaluationAnswer[];
  wrong_answers: EvaluationAnswer[];
  identified_errors: string[];
  needs_reteaching: boolean;
  overall_feedback?: string | null;
}

export interface TutorSessionResponse {
  thread_id: string;
  status: SessionStatus;
  tutor_response?: string | null;
  route?: string | null;
  route_reason?: string | null;
  complexity_score?: number | null;
  target_skill?: string | null;
  memory_classified_skill?: string | null;
  memory_skill_source?: "memory_classifier" | "explicit_target" | "gemini_fallback" | null;
  memory_learner_state?: "NEEDS_SUPPORT" | "DEVELOPING" | "STRONG" | "UNAVAILABLE" | null;
  student_response_message?: string | null;
  assessment_message?: string | null;
  pedagogical_move_message?: string | null;
  allowed_pedagogical_moves: string[];
  questions: AssessmentQuestion[];
  evaluation?: EvaluationResult | null;
  reteach_round: number;
  turn_count: number;
}

export interface LearnerHistoryItem {
  topic: string;
  event_type: string;
  details: Record<string, unknown>;
}

export interface TutorDialogueTurn {
  role: "teacher" | "student";
  content: string;
}

export interface TutorStartPayload {
  student_id?: string;
  age: number;
  question: string;
  topic: string;
  subtopic?: string | null;
  target_skill?: string | null;
  relevant_history: LearnerHistoryItem[];
  previous_errors: string[];
  previous_strategies: string[];
  continuation?: boolean;
  conversation_history?: TutorDialogueTurn[];
}

export interface TutorPracticeStartPayload {
  target_skill: string;
}

export interface TutorPracticeStartResponse {
  target_skill: string;
  problem: string;
  session: TutorSessionResponse;
}

export interface TutorStudentTurnPayload {
  response: string;
}

export interface StudentAnswerPayload {
  question_id: string;
  answer: string;
}

export interface TutorAnswerPayload {
  answers: StudentAnswerPayload[];
}

export interface LearnerSetup {
  studentId: string;
  age: number;
  topic: string;
  subtopic: string;
  targetSkill: string;
  targetSkillSource?: "manual" | null;
}

export interface MoveSelectorContextFeature {
  index: number;
  block: "S" | "K" | "L" | "H" | "Q" | "unknown";
  name: string;
  description: string;
  value: number;
  missing_indicator: boolean;
}

export interface MoveSelectorRewardUpdate {
  action_event_id?: string | null;
  resolution_status?: string | null;
  outcome_observed: boolean;
  mastery_before?: number | null;
  mastery_after?: number | null;
  raw_delta?: number | null;
  reward_mode?: string | null;
  reward_value?: number | null;
  attributed_move?: string | null;
  posterior_updated: boolean;
  posterior_update_weight?: number | null;
  posterior_total_updates_before?: number | null;
  posterior_total_updates_after?: number | null;
  update_id?: string | null;
  update_timestamp?: string | null;
}

export interface MoveSelectorTrace {
  action_event_id?: string | null;
  action_turn_index?: number | null;
  policy_mode?: string | null;
  behavior_policy?: string | null;
  decision_source?: string | null;
  context_schema_id?: string | null;
  context_dimension: number;
  enabled_blocks: string[];
  context_features: MoveSelectorContextFeature[];
  md7_probabilities: Record<string, number>;
  available_arms: string[];
  sampled_scores: Record<string, number>;
  policy_scores: Record<string, number>;
  base_model_move?: string | null;
  selected_move?: string | null;
  tutor_move?: string | null;
  post_selection_override: boolean;
  response_quality_scores: Record<string, number>;
  reward_mode?: string | null;
  latest_reward_update?: MoveSelectorRewardUpdate | null;
}

export interface BKTParametersTrace {
  population_prior?: number | null;
  learn?: number | null;
  guess?: number | null;
  slip?: number | null;
  forget?: number | null;
}

export interface BKTHistoryTrace {
  observations_before_attempt?: number | null;
  dialogue_turns_before_assessment?: number | null;
  dialogue_observations_before_assessment?: number | null;
  observations_before_update?: number | null;
  observations_applied: number;
  observations_after_update?: number | null;
  persisted_suffix_matches?: boolean | null;
}

export interface BKTMasteryTrace {
  before?: number | null;
  after?: number | null;
  delta?: number | null;
  recomputed_before?: number | null;
  before_assessment?: number | null;
  dialogue_delta_before_assessment?: number | null;
  assessment_delta?: number | null;
  recomputed_after?: number | null;
  before_consistent?: boolean | null;
  after_consistent?: boolean | null;
}

export interface BKTEvidenceTrace {
  correctness?: string | null;
  evidence_category?: string | null;
  evaluator_source?: string | null;
  reported_evaluator_confidence?: number | null;
  applied_evaluator_confidence?: number | null;
  evaluator_confidence_cap?: number | null;
}

export interface BKTBehaviourTrace {
  reasoning_probability?: number | null;
  reasoning_present?: boolean | null;
  uncertainty_probability?: number | null;
  uncertainty_present?: boolean | null;
  clarification_probability?: number | null;
  clarification_present?: boolean | null;
  repeated_misunderstanding?: boolean | null;
}

export interface BKTCalculationTrace {
  mastery_before: number;
  likelihood_if_known: number;
  likelihood_if_not_known: number;
  bayes_numerator: number;
  bayes_denominator: number;
  full_observation_posterior: number;
  confidence_weighted_posterior: number;
  effective_learn: number;
  after_learning: number;
  after_forgetting: number;
  recorded_mastery_after: number;
  matches_recorded_mastery: boolean;
}

export interface BKTObservationTrace {
  sequence: number;
  event_id?: string | null;
  source_action_event_id?: string | null;
  skill?: string | null;
  primary_signal?: string | null;
  resolver_version?: string | null;
  should_update: boolean;
  outcome?: number | null;
  evidence_weight?: number | null;
  evaluator_confidence?: number | null;
  behavioural_confidence?: number | null;
  behaviour_factor?: number | null;
  update_confidence?: number | null;
  observation_source?: string | null;
  contributors: string[];
  evidence: BKTEvidenceTrace;
  behaviour: BKTBehaviourTrace;
  persistence_status?: string | null;
  mastery_before?: number | null;
  mastery_after?: number | null;
  delta_mastery?: number | null;
  calculation?: BKTCalculationTrace | null;
}

export interface BKTUpdateTrace {
  schema_version?: string | null;
  event_type?: string | null;
  recorded_at?: string | null;
  attempt_id?: string | null;
  session_id?: string | null;
  target_skill?: string | null;
  turn_index?: number | null;
  update_timing?: string | null;
  assessment_answer_count?: number | null;
  assessment_correct_count?: number | null;
  assessment_incorrect_count?: number | null;
  parameters: BKTParametersTrace;
  effective_initial_prior?: number | null;
  effective_initial_prior_source?: string | null;
  history: BKTHistoryTrace;
  mastery: BKTMasteryTrace;
  observations: BKTObservationTrace[];
  persisted_mastery_probability?: number | null;
  persisted_mastery_label?: string | null;
  persisted_initial_prior?: number | null;
  persisted_initial_prior_source?: string | null;
  compatibility_rebased?: boolean | null;
}

export interface TutorDemoTraceResponse {
  thread_id: string;
  phase: string;
  question: string;
  age: number;
  topic: string;
  subtopic?: string | null;
  target_skill?: string | null;

  complexity_model: string;
  complexity_input_scope: string;
  complexity_score?: number | null;

  relevant_history_count: number;
  previous_error_count: number;
  previous_strategy_count: number;

  route?: string | null;
  route_reason?: string | null;
  planner_used: boolean;
  planner_output_available: boolean;

  pedagogical_move?: string | null;
  selector?: string | null;
  selected_arm?: string | null;
  base_move?: string | null;
  move_overridden?: boolean | null;

  verified_math_ready: boolean;
  verified_math_evidence_count: number;
  tutor_generation_fallback_used?: boolean | null;

  teaching_status?: string | null;
  teaching_progress_reason?: string | null;

  assessment_question_count: number;
  evaluation_available: boolean;
  correct_answer_count: number;
  wrong_answer_count: number;
  needs_reteaching: boolean;

  turn_count: number;
  reteach_round: number;
  attempt_id?: string | null;
  adaptive_attempt_index?: number | null;
  checkpoint_saved: boolean;
  next_nodes: string[];
  move_selector?: MoveSelectorTrace | null;
  bkt_update?: BKTUpdateTrace | null;
}

export interface DemoRouterScenario {
  label: string;
  age: number;
  previous_errors: string[];
  previous_strategies: string[];
}

export interface DemoRouterComparisonPayload {
  question: string;
  topic: string;
  subtopic?: string | null;
  scenarios: DemoRouterScenario[];
}

export interface DemoRouterScenarioResult {
  label: string;
  age: number;
  previous_error_count: number;
  previous_strategy_count: number;
  route: string;
  route_reason: string;
}

export interface DemoRouterComparisonResponse {
  question: string;
  complexity_score: number;
  complexity_input_scope: string;
  scenarios: DemoRouterScenarioResult[];
}
