import type { StrategyInfo } from '../api/types'

interface Props {
  strategies: StrategyInfo[]
  selected: string[]
  params: Record<string, Record<string, unknown>>
  onChange: (selected: string[], params: Record<string, Record<string, unknown>>) => void
}

export default function StrategyConfigForm({ strategies, selected, params, onChange }: Props) {
  const toggleStrategy = (name: string) => {
    const next = selected.includes(name)
      ? selected.filter((s) => s !== name)
      : [...selected, name]
    onChange(next, params)
  }

  const updateParam = (strategy: string, key: string, value: string) => {
    const parsed = value === '' ? '' : isNaN(Number(value)) ? value : Number(value)
    onChange(selected, {
      ...params,
      [strategy]: { ...params[strategy], [key]: parsed },
    })
  }

  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-slate-300">Strategies</label>
      {strategies.map((s) => {
        const isSelected = selected.includes(s.name)
        return (
          <div
            key={s.name}
            className={`rounded-lg border p-4 transition-colors ${
              isSelected
                ? 'border-blue-500 bg-blue-500/5'
                : 'border-slate-700 bg-slate-800/50'
            }`}
          >
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={isSelected}
                onChange={() => toggleStrategy(s.name)}
                className="rounded border-slate-600 bg-slate-700 text-blue-500 focus:ring-blue-500 focus:ring-offset-0"
              />
              <span className="font-medium text-sm">{s.name}</span>
            </label>
            {isSelected && Object.keys(s.default_params ?? {}).length > 0 && (
              <div className="mt-3 ml-7 grid grid-cols-2 gap-3">
                {Object.entries(s.default_params ?? {}).map(([k, defaultVal]) => (
                  <div key={k}>
                    <label className="block text-xs text-slate-400 mb-1">{k}</label>
                    <input
                      type="text"
                      value={String(params[s.name]?.[k] ?? defaultVal ?? '')}
                      onChange={(e) => updateParam(s.name, k, e.target.value)}
                      className="w-full px-3 py-1.5 text-sm bg-slate-700 border border-slate-600 rounded-md text-slate-200 focus:outline-none focus:ring-1 focus:ring-blue-500"
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
