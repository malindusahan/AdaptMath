import React, { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { getHealth, getReadiness } from '../api/memoryApi'
import { HealthResponse, ReadinessResponse } from '../types/api'

export const AppLayout: React.FC = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [healthStatus, setHealthStatus] = useState<HealthResponse | null>(null)
  const [readyStatus, setReadyStatus] = useState<ReadinessResponse | null>(null)
  const [isHealthLoading, setIsHealthLoading] = useState(true)
  const location = useLocation()

  useEffect(() => {
    let isMounted = true

    const checkStatus = async () => {
      try {
        const [h, r] = await Promise.allSettled([getHealth(), getReadiness()])
        if (!isMounted) return

        if (h.status === 'fulfilled') {
          setHealthStatus(h.value)
        } else {
          setHealthStatus(null)
        }

        if (r.status === 'fulfilled') {
          setReadyStatus(r.value)
        } else {
          setReadyStatus(null)
        }
      } catch {
        if (isMounted) {
          setHealthStatus(null)
          setReadyStatus(null)
        }
      } finally {
        if (isMounted) {
          setIsHealthLoading(false)
        }
      }
    }

    checkStatus()
    const interval = setInterval(checkStatus, 15000)

    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, [])

  const navItems = [
    {
      to: '/',
      label: 'Student Demo',
      description: 'Interactive question analysis & context engine',
      icon: (
        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
      ),
    },
  ]

  const isOnline = healthStatus?.status === 'ok'
  const isReady = readyStatus?.status === 'ready'

  return (
    <div className="min-h-screen flex flex-col bg-slate-100/70 text-slate-800 antialiased font-sans">
      {/* Top Academic Header */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Mobile menu trigger */}
            <button
              type="button"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-1.5 rounded text-slate-500 hover:text-slate-900 hover:bg-slate-100 focus:outline-none focus:ring-2 focus:ring-slate-400"
              aria-label="Toggle navigation menu"
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                {mobileMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>

            <div className="flex items-center gap-2.5">
              <span className="w-6 h-6 rounded bg-slate-900 text-white flex items-center justify-center font-mono font-bold text-xs">
                M
              </span>
              <div className="leading-none">
                <span className="text-sm font-bold text-slate-900 tracking-tight block">
                  Student Personalization Memory
                </span>
                <span className="text-[11px] text-slate-500 font-normal">
                  Multi-Agent Cognitive & Epistemic Engine
                </span>
              </div>
            </div>
          </div>

          {/* Backend Status Indicators */}
          <div className="flex items-center gap-2.5 text-xs">
            {/* Service Liveness */}
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border font-mono text-[11px] transition-colors ${
                isHealthLoading
                  ? 'bg-slate-50 text-slate-500 border-slate-200'
                  : isOnline
                  ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                  : 'bg-rose-50 text-rose-700 border-rose-200'
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  isHealthLoading
                    ? 'bg-slate-400 animate-pulse'
                    : isOnline
                    ? 'bg-emerald-500'
                    : 'bg-rose-500'
                }`}
              />
              Service: {isHealthLoading ? 'Checking...' : isOnline ? 'Online' : 'Offline'}
            </span>

            {/* Dependency Readiness */}
            <span
              className={`inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full border font-mono text-[11px] transition-colors ${
                isHealthLoading
                  ? 'bg-slate-50 text-slate-500 border-slate-200'
                  : isReady
                  ? 'bg-sky-50 text-sky-700 border-sky-200'
                  : 'bg-amber-50 text-amber-700 border-amber-200'
              }`}
            >
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  isHealthLoading
                    ? 'bg-slate-400 animate-pulse'
                    : isReady
                    ? 'bg-sky-500'
                    : 'bg-amber-500'
                }`}
              />
              Readiness: {isHealthLoading ? 'Checking...' : isReady ? 'Ready' : 'Not Ready'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Dashboard Container */}
      <div className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 flex flex-col md:flex-row gap-6">
        {/* Sidebar Navigation */}
        <aside
          className={`md:w-56 shrink-0 ${
            mobileMenuOpen ? 'block' : 'hidden md:block'
          }`}
        >
          <div className="sticky top-20 space-y-4">
            <nav className="bg-white rounded-lg border border-slate-200 p-2 shadow-xs space-y-1">
              <div className="px-3 py-2 text-[10px] font-bold uppercase tracking-wider text-slate-400">
                Navigation
              </div>
              {navItems.map((item) => {
                const isActive = location.pathname === item.to || (item.to !== '/' && location.pathname.startsWith(item.to))
                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    onClick={() => setMobileMenuOpen(false)}
                    className={`flex items-center gap-2.5 px-3 py-2 rounded-md text-xs font-medium transition-colors ${
                      isActive
                        ? 'bg-slate-900 text-white shadow-xs'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-50'
                    }`}
                  >
                    <span className={isActive ? 'text-white' : 'text-slate-400'}>
                      {item.icon}
                    </span>
                    <span>{item.label}</span>
                  </NavLink>
                )
              })}
            </nav>

            <div className="p-3 bg-white rounded-lg border border-slate-200 shadow-xs text-xs text-slate-500 space-y-1.5">
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                Architecture
              </div>
              <div className="text-[11px] leading-relaxed text-slate-600">
                PostgreSQL • MiniLM Centroid Extractor • Logistic Classifier • 3-Tier Memory
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content Area */}
        <main className="flex-1 min-w-0">
          <Outlet />
        </main>
      </div>

      {/* Research Footer */}
      <footer className="mt-auto border-t border-slate-200 bg-white py-3">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-400">
          <div>Student Personalization Memory Service — Research Edition</div>
          <div className="font-mono text-[11px]">Schema: student_memory • v1.0.0</div>
        </div>
      </footer>
    </div>
  )
}

export default AppLayout
