import { Suspense } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import Spinner from './Spinner'

const backtestLinks = [
  { to: '/', label: 'Dashboard', icon: '📊' },
  { to: '/new', label: 'New Backtest', icon: '🚀' },
  { to: '/inspector', label: 'Agent Inspector', icon: '🔍' },
]

const liveLinks = [
  { to: '/live', label: 'Live Dashboard', icon: '⚡' },
  { to: '/live/approvals', label: 'Approvals', icon: '✓' },
  { to: '/live/orders', label: 'Orders', icon: '↹' },
  { to: '/live/config', label: 'Configuration', icon: '⚙' },
  { to: '/live/config#ai', label: 'AI Settings', icon: '🧠' },
]

function SidebarLink({ to, label, icon, end }: { to: string; label: string; icon: string; end?: boolean }) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
          isActive
            ? 'bg-blue-600/20 text-blue-400'
            : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
        }`
      }
    >
      <span className="text-base">{icon}</span>
      {label}
    </NavLink>
  )
}

export default function Layout() {
  return (
    <div className="flex h-screen bg-slate-900 text-slate-200">
      <aside className="w-60 flex-shrink-0 bg-slate-950 border-r border-slate-700 flex flex-col">
        <div className="px-5 py-5 border-b border-slate-700">
          <h1 className="text-lg font-bold tracking-tight text-white">
            AI Trading System
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">Multi-Agent Trading Platform</p>
        </div>
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {backtestLinks.map((l) => (
            <SidebarLink key={l.to} to={l.to} label={l.label} icon={l.icon} end={l.to === '/'} />
          ))}

          <div className="border-t border-slate-700 my-3" />
          <p className="px-3 py-1 text-xs font-semibold text-slate-500 uppercase tracking-wider">
            Live Trading
          </p>

          {liveLinks.map((l) => (
            <SidebarLink key={l.to} to={l.to} label={l.label} icon={l.icon} end={l.to === '/live'} />
          ))}
        </nav>
        <div className="px-5 py-4 border-t border-slate-700 text-xs text-slate-500">
          v0.1.0
        </div>
      </aside>

      <main className="flex-1 overflow-auto">
        <div className="p-6 max-w-[1600px] mx-auto">
          <Suspense
            fallback={
              <div className="flex items-center justify-center h-64">
                <Spinner className="h-8 w-8" />
              </div>
            }
          >
            <Outlet />
          </Suspense>
        </div>
      </main>
    </div>
  )
}
