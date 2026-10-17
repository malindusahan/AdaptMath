import apiClient from './client'
import {
  LoginRequest,
  LoginResponse,
  SignupRequest,
  SignupResponse,
  UserMeResponse,
} from '../types/api'

export const signup = async (payload: SignupRequest): Promise<SignupResponse> => {
  const response = await apiClient.post<SignupResponse>('/auth/signup', payload)
  return response.data
}

export const login = async (payload: LoginRequest): Promise<LoginResponse> => {
  const response = await apiClient.post<LoginResponse>('/auth/login', payload)
  const data = response.data
  if (data.token) {
    localStorage.setItem('mem_session_token', data.token)
    localStorage.setItem('mem_auth_user', JSON.stringify(data))
  }
  return data
}

export const getMe = async (): Promise<UserMeResponse> => {
  const response = await apiClient.get<UserMeResponse>('/auth/me')
  return response.data
}

export const logout = async (): Promise<void> => {
  try {
    await apiClient.post('/auth/logout')
  } finally {
    localStorage.removeItem('mem_session_token')
    localStorage.removeItem('mem_auth_user')
  }
}

export const getStoredAuthUser = (): LoginResponse | null => {
  try {
    const raw = localStorage.getItem('mem_auth_user')
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}
