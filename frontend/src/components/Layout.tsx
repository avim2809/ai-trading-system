import { Suspense, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import Spinner from './Spinner'
import { getInstance, setInstance, type LiveInstance } from '../api/client'

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

const monitoringLinks = [
  { to: '/logs', label: 'Live Logs', icon: '📜' },
  { to: '/decisions', label: 'Decision Log', icon: '🧠' },
  { to: '/system', label: 'Server Resources', icon: '🖥' },
]

function SidebarLink({
  to,
  label,
  icon,
  end,
  onNavigate,
}: {
  to: string
  label: string
  icon: string
  end?: boolean
  onNavigate?: () => void
}) {
  return (
    <NavLink
      to={to}
      end={end}
      onClick={onNavigate}
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

function InstanceSwitcher({ className = '' }: { className?: string }) {
  const current = getInstance()

  const pick = (instance: LiveInstance) => {
    if (instance !== current) setInstance(instance)
  }

  const optionClass = (instance: LiveInstance) =>
    `flex-1 px-2 py-1.5 text-xs font-semibold rounded-md transition-colors ${
      current === instance
        ? 'bg-blue-600 text-white'
        : 'text-slate-400 hover:text-slate-200'
    }`

  return (
    <div className={`flex gap-1 p-1 bg-slate-800 rounded-lg ${className}`}>
      <button onClick={() => pick('ibkr')} className={optionClass('ibkr')}>
        IBKR
      </button>
      <button onClick={() => pick('alpaca')} className={optionClass('alpaca')}>
        Alpaca
      </button>
    </div>
  )
}

export default function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const closeMobile = () => setMobileOpen(false)

  return (
    <div className="flex h-screen bg-slate-900 text-slate-200">
      {/* Backdrop, mobile only, shown while the nav drawer is open */}
      {mobileOpen && (
        <div
          className="fixed inset-0 bg-black/60 z-30 md:hidden"
          onClick={closeMobile}
          aria-hidden="true"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 w-64 flex-shrink-0 bg-slate-950 border-r border-slate-700 flex flex-col transform transition-transform duration-200 ease-in-out md:static md:z-auto md:translate-x-0 ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="px-5 py-5 border-b border-slate-700 flex items-start justify-between">
          <div>
            <h1 className="text-lg font-bold tracking-tight text-white">
              AI Trading System
            </h1>
            <p className="text-xs text-slate-400 mt-0.5">Multi-Agent Trading Platform</p>
          </div>
          <button
            onClick={closeMobile}
            className="md:hidden text-slate-400 hover:text-slate-200 p-1 -mr-1 -mt-1"
            aria-label="Close menu"
          >
            ✕
          </button>
        </div>
        <div className="px-5 pt-4">
          <InstanceSwitcher />
        </div>
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {backtestLinks.map((l) => (
            <SidebarLink key={l.to} to={l.to} label={l.label} icon={l.icon} end={l.to === '/'} onNavigate={closeMobile} />
          ))}

          <div className="border-t border-slate-700 my-3" />
          <p className="px-3 py-1 text-xs font-semibold text-slate-500 uppercase tracking-wider">
            Live Trading
          </p>

          {liveLinks.map((l) => (
            <SidebarLink key={l.to} to={l.to} label={l.label} icon={l.icon} end={l.to === '/live'} onNavigate={closeMobile} />
          ))}

          <div className="border-t border-slate-700 my-3" />
          <p className="px-3 py-1 text-xs font-semibold text-slate-500 uppercase tracking-wider">
            Monitoring
          </p>

          {monitoringLinks.map((l) => (
            <SidebarLink key={l.to} to={l.to} label={l.label} icon={l.icon} onNavigate={closeMobile} />
          ))}
        </nav>
        <div className="px-5 py-4 border-t border-slate-700 text-xs text-slate-500">
          v0.1.0
        </div>
      </aside>

      <div className="flex-1 flex flex-col min-w-0">
        {/* Mobile-only top bar with the nav-drawer toggle */}
        <header className="md:hidden flex items-center gap-3 px-4 py-3 bg-slate-950 border-b border-slate-700 flex-shrink-0">
          <button
            onClick={() => setMobileOpen(true)}
            className="text-slate-300 hover:text-white p-1 -ml-1 text-xl leading-none"
            aria-label="Open menu"
          >
            ☰
          </button>
          <h1 className="text-sm font-bold text-white flex-1">AI Trading System</h1>
          <InstanceSwitcher className="w-32 flex-shrink-0" />
        </header>

        <main className="flex-1 overflow-auto">
          <div className="p-4 md:p-6 max-w-[1600px] mx-auto">
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
    </div>
  )
}
