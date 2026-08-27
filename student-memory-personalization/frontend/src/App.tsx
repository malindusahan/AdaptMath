import React from 'react'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { getStoredAuthUser } from './api/authApi'
import ChatPage from './pages/ChatPage'
import LoginPage from './pages/LoginPage'
import SignupPage from './pages/SignupPage'

const RootRedirect: React.FC = () => {
  const user = getStoredAuthUser()
  if (!user) {
    return <Navigate to="/login" replace />
  }
  return <Navigate to="/chat" replace />
}

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<RootRedirect />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/chat" element={<ChatPage />} />
        <Route path="/student" element={<ChatPage />} />
        <Route path="*" element={<RootRedirect />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
