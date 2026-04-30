import apiClient from './client'
import {
  AssessmentMemoryUpdateRequest,
  FaprContextResponse,
  HealthResponse,
  MemoryUpdateResponse,
  MetaSignalsResponse,
  PlannerContextResponse,
  QuestionContextRequest,
  QuestionContextResponse,
  ReadinessResponse,
  RepairOutcomeCreateRequest,
  RepairOutcomeResponse,
  SessionItemResponse,
  StudentContextResponse,
  SupportPreferenceResponse,
  TutorContextResponse,
  SkillItem,
} from '../types/api'

export const getHealth = async (): Promise<HealthResponse> => {
  const response = await apiClient.get<HealthResponse>('/health')
  return response.data
}

export const getReadiness = async (): Promise<ReadinessResponse> => {
  const response = await apiClient.get<ReadinessResponse>('/ready')
  return response.data
}

export const createSession = async (): Promise<SessionItemResponse> => {
  const response = await apiClient.post<SessionItemResponse>('/ui/sessions')
  return response.data
}

export const listSessions = async (limit = 30): Promise<SessionItemResponse[]> => {
  const response = await apiClient.get<SessionItemResponse[]>('/ui/sessions', {
    params: { limit },
  })
  return response.data
}

export const updateSession = async (
  sessionId: string,
  title: string
): Promise<SessionItemResponse> => {
  const response = await apiClient.patch<SessionItemResponse>(
    `/ui/sessions/${encodeURIComponent(sessionId)}`,
    { title }
  )
  return response.data
}

export const getQuestionContext = async (
  payload: QuestionContextRequest
): Promise<QuestionContextResponse> => {
  const response = await apiClient.post<QuestionContextResponse>(
    '/ui/question-context',
    payload
  )
  return response.data
}

export const submitMemoryUpdate = async (
  payload: AssessmentMemoryUpdateRequest
): Promise<MemoryUpdateResponse> => {
  const response = await apiClient.post<MemoryUpdateResponse>(
    '/ui/memory-update',
    payload
  )
  return response.data
}

export const submitRepairOutcome = async (
  payload: RepairOutcomeCreateRequest
): Promise<RepairOutcomeResponse> => {
  const response = await apiClient.post<RepairOutcomeResponse>(
    '/ui/repair-outcome',
    payload
  )
  return response.data
}

export interface ContextParams {
  session_id?: string
  skill_id?: string
  limit?: number
  offset?: number
}

export const getStudentContext = async (
  studentId: string,
  params?: ContextParams
): Promise<StudentContextResponse> => {
  const response = await apiClient.get<StudentContextResponse>(
    `/ui/${encodeURIComponent(studentId)}/context`,
    { params }
  )
  return response.data
}

export const getTutorContext = async (
  studentId: string,
  params?: ContextParams
): Promise<TutorContextResponse> => {
  const response = await apiClient.get<TutorContextResponse>(
    `/ui/${encodeURIComponent(studentId)}/tutor-context`,
    { params }
  )
  return response.data
}

export const getPlannerContext = async (
  studentId: string,
  params?: ContextParams
): Promise<PlannerContextResponse> => {
  const response = await apiClient.get<PlannerContextResponse>(
    `/ui/${encodeURIComponent(studentId)}/planner-context`,
    { params }
  )
  return response.data
}

export const getFaprContext = async (
  studentId: string,
  params?: ContextParams
): Promise<FaprContextResponse> => {
  const response = await apiClient.get<FaprContextResponse>(
    `/ui/${encodeURIComponent(studentId)}/fapr-context`,
    { params }
  )
  return response.data
}

export const getSupportPreference = async (
  studentId: string,
  params?: { skill_id?: string }
): Promise<SupportPreferenceResponse> => {
  const response = await apiClient.get<SupportPreferenceResponse>(
    `/ui/${encodeURIComponent(studentId)}/support-preference`,
    { params }
  )
  return response.data
}

export const getMetaSignals = async (
  studentId: string,
  params?: ContextParams
): Promise<MetaSignalsResponse> => {
  const response = await apiClient.get<MetaSignalsResponse>(
    `/ui/${encodeURIComponent(studentId)}/meta-signals`,
    { params }
  )
  return response.data
}

export const listSkills = async (): Promise<SkillItem[]> => {
  const response = await apiClient.get<SkillItem[]>('/ui/skills')
  return response.data
}
