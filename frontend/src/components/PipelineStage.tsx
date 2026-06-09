import type { ReactNode } from 'react'

interface Props {
  title: string
  children: ReactNode
  status?: 'active' | 'complete'
}

export default function PipelineStage({ title, children, status }: Props) {
  const borderColor =
    status === 'active'
      ? 'border-blue-500'
      : status === 'complete'
        ? 'border-emerald-600'
        : 'border-slate-700'

  const dotColor =
    status === 'active'
      ? 'bg-blue-500'
      : status === 'complete'
        ? 'bg-emerald-500'
        : 'bg-slate-600'

  return (
    <div className="relative">
      <div className="absolute left-5 top-0 bottom-0 w-px bg-slate-700 -z-10" />
      <div className={`bg-slate-800 rounded-xl border-l-4 ${borderColor} p-5`}>
        <div className="flex items-center gap-3 mb-3">
          <div className={`w-3 h-3 rounded-full ${dotColor} flex-shrink-0`} />
          <h3 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">
            {title}
          </h3>
        </div>
        <div className="ml-6">{children}</div>
      </div>
    </div>
  )
}
