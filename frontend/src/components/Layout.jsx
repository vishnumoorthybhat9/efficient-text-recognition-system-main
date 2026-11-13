import { useEffect, useState } from 'react'
import { NavLink, useLocation } from 'react-router-dom'
import {
  Activity,
  BarChart3,
  Cpu,
  Database,
  FlaskConical,
  History,
  Info,
  Layers,
  Menu,
  Moon,
  ScanText,
  Sliders,
  Sun,
  X,
} from 'lucide-react'
import { api } from '../api/client.js'
import { useAsync } from '../hooks/useAsync.js'
import { Badge } from './ui.jsx'
import { cx } from '../lib/format.js'

const NAVIGATION = [
  { to: '/', label: 'Dashboard', icon: Activity, end: true },
  { to: '/recognize', label: 'Recognise', icon: ScanText },
  { to: '/preprocess', label: 'Preprocessing Lab', icon: Sliders },
  { to: '/compare', label: 'Compare models', icon: Layers },
  { to: '/evaluation', label: 'Evaluation', icon: BarChart3 },
  { to: '/training', label: 'Training', icon: FlaskConical },
  { to: '/datasets', label: 'Datasets', icon: Database },
  { to: '/history', label: 'History', icon: History },
  { to: '/about', label: 'About', icon: Info },
]

function ThemeToggle() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))

  const toggle = () => {
    const next = !dark
    setDark(next)
    document.documentElement.classList.toggle('dark', next)
    localStorage.setItem('theme', next ? 'dark' : 'light')
  }

  return (
    <button
      onClick={toggle}
      aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
      className="rounded-lg p-2 text-slate-500 transition-colors hover:bg-slate-100 hover:text-slate-900 dark:hover:bg-slate-800 dark:hover:text-slate-100"
    >
      {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
    </button>
  )
}

/** Live backend status, so a stopped server is obvious rather than confusing. */
function StatusPill() {
  const { data, error, run } = useAsync(() => api.health(), [])

  useEffect(() => {
    const handle = setInterval(run, 20000)
    return () => clearInterval(handle)
  }, [run])

  if (error) {
    return (
      <div className="flex items-center gap-2 rounded-lg bg-rose-50 px-2.5 py-1.5 dark:bg-rose-500/10">
        <span className="h-2 w-2 shrink-0 rounded-full bg-rose-500" />
        <span className="text-[11px] font-medium text-rose-700 dark:text-rose-300">
          Backend offline
        </span>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="flex items-center gap-2 rounded-lg px-2.5 py-1.5">
        <span className="h-2 w-2 shrink-0 animate-pulse rounded-full bg-slate-400" />
        <span className="text-[11px] text-slate-500">Connecting…</span>
      </div>
    )
  }

  const healthy = data.status === 'ok'
  return (
    <div
      className={cx(
        'flex items-center gap-2 rounded-lg px-2.5 py-1.5',
        healthy ? 'bg-emerald-50 dark:bg-emerald-500/10' : 'bg-amber-50 dark:bg-amber-500/10',
      )}
      title={`${data.models_available} model(s) available · database ${data.database_ready ? 'ready' : 'unavailable'}`}
    >
      <span className={cx('h-2 w-2 shrink-0 rounded-full', healthy ? 'bg-emerald-500' : 'bg-amber-500')} />
      <span
        className={cx(
          'text-[11px] font-medium',
          healthy ? 'text-emerald-700 dark:text-emerald-300' : 'text-amber-700 dark:text-amber-300',
        )}
      >
        {healthy ? 'Ready' : 'Degraded'}
      </span>
      <Badge tone={data.device?.startsWith('cuda') ? 'indigo' : 'slate'}>
        <Cpu className="mr-1 h-3 w-3" />
        {data.device?.toUpperCase() || 'CPU'}
      </Badge>
    </div>
  )
}

export function Layout({ children }) {
  const [open, setOpen] = useState(false)
  const location = useLocation()

  // Close the mobile drawer whenever navigation happens, otherwise it covers
  // the page the user just asked for.
  useEffect(() => {
    setOpen(false)
  }, [location.pathname])

  return (
    <div className="flex h-full">
      {/* Mobile backdrop */}
      {open && (
        <div
          className="fixed inset-0 z-30 bg-slate-900/50 lg:hidden"
          onClick={() => setOpen(false)}
          aria-hidden="true"
        />
      )}

      <aside
        className={cx(
          'fixed inset-y-0 left-0 z-40 flex w-64 shrink-0 flex-col border-r border-slate-200 bg-white transition-transform lg:static lg:translate-x-0',
          'dark:border-slate-800 dark:bg-slate-900',
          open ? 'translate-x-0' : '-translate-x-full',
        )}
      >
        <div className="flex items-center gap-2.5 border-b border-slate-200 px-5 py-4 dark:border-slate-800">
          <span className="rounded-lg bg-indigo-600 p-1.5 text-white">
            <ScanText className="h-4 w-4" />
          </span>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-semibold">Text Recognition</p>
            <p className="truncate text-[10px] text-slate-500">Deep + Self-Supervised Learning</p>
          </div>
          <button
            className="rounded-lg p-1 text-slate-500 lg:hidden"
            onClick={() => setOpen(false)}
            aria-label="Close navigation"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <nav className="flex-1 space-y-0.5 overflow-y-auto p-3">
          {NAVIGATION.map(({ to, label, icon: Icon, end }) => (
            <NavLink
              key={to}
              to={to}
              end={end}
              className={({ isActive }) =>
                cx(
                  'flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors',
                  isActive
                    ? 'bg-indigo-50 font-medium text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-300'
                    : 'text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800',
                )
              }
            >
              <Icon className="h-4 w-4 shrink-0" />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="border-t border-slate-200 p-3 dark:border-slate-800">
          <p className="px-2 text-[10px] leading-relaxed text-slate-400">
            B.Tech Major Project · Project Code 42
            <br />
            Dept. of Information Science &amp; Engineering
          </p>
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-slate-200 bg-white/80 px-4 py-2.5 backdrop-blur dark:border-slate-800 dark:bg-slate-900/80">
          <button
            className="rounded-lg p-2 text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800 lg:hidden"
            onClick={() => setOpen(true)}
            aria-label="Open navigation"
          >
            <Menu className="h-4 w-4" />
          </button>
          <div className="flex-1" />
          <StatusPill />
          <ThemeToggle />
        </header>

        <main className="flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-7xl px-4 py-6 sm:px-6 lg:px-8">{children}</div>
        </main>
      </div>
    </div>
  )
}

export function PageHeader({ title, description, action }) {
  return (
    <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold tracking-tight text-slate-900 dark:text-slate-100">
          {title}
        </h1>
        {description && (
          <p className="mt-1 max-w-3xl text-sm text-slate-500 dark:text-slate-400">{description}</p>
        )}
      </div>
      {action}
    </div>
  )
}
