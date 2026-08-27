import React from 'react'

interface LoadingSpinnerProps {
  label?: string
  size?: 'sm' | 'md' | 'lg'
}

const sizeClasses: Record<'sm' | 'md' | 'lg', string> = {
  sm: 'w-4 h-4 border-2',
  md: 'w-6 h-6 border-2',
  lg: 'w-8 h-8 border-3',
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  label = 'Loading...',
  size = 'md',
}) => {
  return (
    <div className="flex flex-col items-center justify-center gap-2 p-4 text-slate-500">
      <div
        className={`animate-spin rounded-full border-slate-200 border-t-indigo-600 ${sizeClasses[size]}`}
        role="status"
        aria-label={label}
      />
      {label && <span className="text-xs font-medium text-slate-500">{label}</span>}
    </div>
  )
}

export default LoadingSpinner
