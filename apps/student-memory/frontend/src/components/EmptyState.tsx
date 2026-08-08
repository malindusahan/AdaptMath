import React from 'react'

interface EmptyStateProps {
  title?: string
  message: string
  icon?: React.ReactNode
  action?: React.ReactNode
  compact?: boolean
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title,
  message,
  icon,
  action,
  compact = false,
}) => {
  return (
    <div
      className={`text-center flex flex-col items-center justify-center rounded-md border border-dashed border-slate-200 bg-slate-50/40 text-slate-500 ${
        compact ? 'py-6 px-4' : 'py-10 px-6'
      }`}
    >
      {icon && (
        <div className="mb-2 text-slate-400">
          {icon}
        </div>
      )}
      {title && (
        <h3 className="text-sm font-medium text-slate-800 mb-1">{title}</h3>
      )}
      <p className="text-xs text-slate-500 max-w-sm leading-relaxed">{message}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export default EmptyState
