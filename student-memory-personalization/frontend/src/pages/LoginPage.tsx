import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { login } from '../api/authApi'

export const LoginPage: React.FC = () => {
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const quickDemoAccounts = [
    { label: 'demo_new', role: 'STUDENT', desc: 'Cold start, no memory' },
    { label: 'demo_developing', role: 'STUDENT', desc: 'Linear Equations, Developing' },
    { label: 'demo_strong', role: 'STUDENT', desc: 'Pythagorean, Strong' },
    { label: 'demo_support', role: 'STUDENT', desc: 'Repair history & Support preference' },
    { label: 'Student01', role: 'STUDENT', desc: 'Personalized Student' },
  ]

  const handleSelectDemo = (user: string) => {
    setUsername(user)
    setPassword('pass123')
    setErrorMsg(null)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim() || !password) {
      setErrorMsg('Please enter both username and password.')
      return
    }

    setIsLoading(true)
    setErrorMsg(null)

    try {
      await login({
        username: username.trim(),
        password,
      })

      navigate('/chat')
    } catch (err: any) {
      const msg =
        err.response?.data?.message ||
        err.response?.data?.detail ||
        'Login failed. Please check your credentials.'
      setErrorMsg(msg)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-4">
      <div className="w-full max-w-md bg-slate-900/90 backdrop-blur-md border border-slate-800 rounded-2xl shadow-2xl p-8 space-y-6">
        <div className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 mb-2">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 11c0 3.517-1.009 6.799-2.753 9.571m-3.44-2.04l.054-.09A13.916 13.916 0 008 11a4 4 0 118 0c0 1.017-.07 2.019-.203 3m-2.118 6.844A21.88 21.88 0 0015.171 17m3.839 1.132c.645-2.266.99-4.659.99-7.132A8 8 0 004 11a7.96 7.96 0 00.99 3.868" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Student Personalization Memory</h1>
          <p className="text-sm text-slate-400">Sign in to access your continuous learning state</p>
        </div>

        {errorMsg && (
          <div className="p-3.5 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-300 text-sm flex items-start gap-2.5">
            <svg className="w-5 h-5 flex-shrink-0 text-rose-400 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>{errorMsg}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
              Username
            </label>
            <input
              type="text"
              id="input-username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="e.g. demo_developing or student01"
              className="w-full bg-slate-950/80 border border-slate-800 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 transition-all outline-none"
              required
            />
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1.5">
              Password
            </label>
            <input
              type="password"
              id="input-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              className="w-full bg-slate-950/80 border border-slate-800 focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 transition-all outline-none"
              required
            />
          </div>

          <button
            type="submit"
            id="btn-login"
            disabled={isLoading}
            className="w-full mt-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white font-medium py-2.5 rounded-xl transition-all shadow-lg shadow-indigo-600/25 flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              'Sign In'
            )}
          </button>
        </form>

        <div className="space-y-3 pt-2">
          <div className="flex items-center justify-between text-xs text-slate-400 border-t border-slate-800/80 pt-4">
            <span>Quick Demo Accounts:</span>
            <span className="text-slate-500">password: pass123</span>
          </div>

          <div className="grid grid-cols-2 gap-2">
            {quickDemoAccounts.map((acc) => (
              <button
                key={acc.label}
                type="button"
                onClick={() => handleSelectDemo(acc.label)}
                className={`text-left p-2 rounded-lg border text-xs transition-all ${
                  username === acc.label
                    ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300 font-semibold'
                    : 'bg-slate-950/50 border-slate-800 hover:border-slate-700 text-slate-300'
                }`}
              >
                <div className="font-mono">{acc.label}</div>
                <div className="text-[10px] text-slate-500 truncate">{acc.desc}</div>
              </button>
            ))}
          </div>
        </div>

        <div className="text-center text-xs text-slate-400 pt-2">
          Don't have an account?{' '}
          <Link to="/signup" className="text-indigo-400 hover:text-indigo-300 font-medium underline underline-offset-4">
            Create Student Account
          </Link>
        </div>
      </div>
    </div>
  )
}

export default LoginPage
