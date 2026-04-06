import React from 'react'

interface SectionCardProps {
  title: string
  subtitle?: string
  badge?: React.ReactNode
  action?: React.ReactNode
  children: React.ReactNode
  className?: string
  headerClassName?: string
}

export const SectionCard: React.FC<SectionCardProps> = ({
  title,
  subtitle,
  badge,
  action,
  children,
  className = '',
  headerClassName = '',
}) => {
  return (
    <div className={`bg-white rounded-lg border border-slate-200 shadow-xs overflow-hidden ${className}`}>
      <div
        className={`px-5 py-3.5 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between gap-3 ${headerClassName}`}
      >
        <div className="flex items-center gap-2.5 min-w-0">
          <h2 className="text-sm font-semibold text-slate-900 tracking-tight truncate">
            {title}
          </h2>
          {badge}
        </div>
        {action && <div className="shrink-0">{action}</div>}
      </div>
      {subtitle && (
        <div className="px-5 pt-3 pb-0 text-xs text-slate-500">{subtitle}</div>
      )}
      <div className="p-5">{children}</div>
    </div>
  )
}

export default SectionCard
