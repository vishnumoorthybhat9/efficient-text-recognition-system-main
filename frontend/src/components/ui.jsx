import { AlertTriangle, CheckCircle2, Info, Loader2, XCircle } from 'lucide-react'
import { cx } from '../lib/format.js'

export function Card({ className, children, ...rest }) {
  return (
    <div className={cx('card', className)} {...rest}>
      {children}
    </div>
  )
}

export function CardHeader({ title, subtitle, action, icon: Icon }) {
  return (
    <div className="flex items-start justify-between gap-4 border-b border-slate-200 px-5 py-4 dark:border-slate-800">
      <div className="flex min-w-0 items-start gap-3">
        {Icon && (
          <span className="mt-0.5 rounded-lg bg-indigo-50 p-2 text-indigo-600 dark:bg-indigo-500/10 dark:text-indigo-400">
            <Icon className="h-4 w-4" />
          </span>
        )}
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold text-slate-900 dark:text-slate-100">{title}</h2>
          {subtitle && (
            <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">{subtitle}</p>
          )}
        </div>
      </div>
      {action && <div className="shrink-0">{action}</div>}
    </div>
  )
}

const BUTTON_VARIANTS = {
  primary:
    'bg-indigo-600 text-white hover:bg-indigo-500 active:bg-indigo-700 disabled:bg-indigo-400',
  secondary:
    'border border-slate-300 bg-white text-slate-700 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:bg-slate-800',
  ghost:
    'text-slate-600 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800',
  danger: 'bg-rose-600 text-white hover:bg-rose-500 active:bg-rose-700 disabled:bg-rose-400',
  success: 'bg-emerald-600 text-white hover:bg-emerald-500 disabled:bg-emerald-400',
}

const BUTTON_SIZES = {
  sm: 'px-2.5 py-1.5 text-xs gap-1.5',
  md: 'px-3.5 py-2 text-sm gap-2',
  lg: 'px-5 py-2.5 text-sm gap-2',
}

export function Button({
  variant = 'primary',
  size = 'md',
  loading = false,
  icon: Icon,
  className,
  children,
  disabled,
  ...rest
}) {
  return (
    <button
      className={cx(
        'inline-flex items-center justify-center rounded-lg font-medium transition-colors',
        'disabled:cursor-not-allowed disabled:opacity-60',
        BUTTON_VARIANTS[variant],
        BUTTON_SIZES[size],
        className,
      )}
      disabled={disabled || loading}
      {...rest}
    >
      {loading ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : (
        Icon && <Icon className="h-4 w-4" />
      )}
      {children}
    </button>
  )
}

const BADGE_TONES = {
  slate: 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300',
  indigo: 'bg-indigo-50 text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-300',
  emerald: 'bg-emerald-50 text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-300',
  amber: 'bg-amber-50 text-amber-700 dark:bg-amber-500/10 dark:text-amber-300',
  rose: 'bg-rose-50 text-rose-700 dark:bg-rose-500/10 dark:text-rose-300',
  blue: 'bg-sky-50 text-sky-700 dark:bg-sky-500/10 dark:text-sky-300',
}

export function Badge({ tone = 'slate', className, children }) {
  return (
    <span
      className={cx(
        'inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium',
        BADGE_TONES[tone],
        className,
      )}
    >
      {children}
    </span>
  )
}

/**
 * Badge showing whether a model was trained here or downloaded.
 *
 * Present on every model throughout the UI: a pretrained baseline's numbers
 * must never be mistaken for a result of this project.
 */
export function ProvenanceBadge({ provenance }) {
  const isOurs = provenance === 'trained_in_project'
  return (
    <Badge tone={isOurs ? 'emerald' : 'amber'}>
      {isOurs ? 'Trained here' : 'Pretrained'}
    </Badge>
  )
}

export function Spinner({ className }) {
  return <Loader2 className={cx('h-5 w-5 animate-spin text-indigo-500', className)} />
}

export function Loading({ label = 'Loading…', className }) {
  return (
    <div className={cx('flex items-center justify-center gap-3 py-12 text-sm text-slate-500', className)}>
      <Spinner />
      <span>{label}</span>
    </div>
  )
}

const ALERT_TONES = {
  info: {
    icon: Info,
    className: 'border-sky-200 bg-sky-50 text-sky-900 dark:border-sky-500/30 dark:bg-sky-500/10 dark:text-sky-200',
    iconClass: 'text-sky-500',
  },
  success: {
    icon: CheckCircle2,
    className:
      'border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-200',
    iconClass: 'text-emerald-500',
  },
  warning: {
    icon: AlertTriangle,
    className:
      'border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200',
    iconClass: 'text-amber-500',
  },
  error: {
    icon: XCircle,
    className:
      'border-rose-200 bg-rose-50 text-rose-900 dark:border-rose-500/30 dark:bg-rose-500/10 dark:text-rose-200',
    iconClass: 'text-rose-500',
  },
}

export function Alert({ tone = 'info', title, children, action, className }) {
  const config = ALERT_TONES[tone] || ALERT_TONES.info
  const Icon = config.icon
  return (
    <div className={cx('flex gap-3 rounded-lg border px-4 py-3 text-sm', config.className, className)}>
      <Icon className={cx('mt-0.5 h-4 w-4 shrink-0', config.iconClass)} />
      <div className="min-w-0 flex-1">
        {title && <p className="font-medium">{title}</p>}
        {children && <div className={cx('text-[13px]', title && 'mt-1 opacity-90')}>{children}</div>}
      </div>
      {action && <div className="shrink-0 self-center">{action}</div>}
    </div>
  )
}

/** Render an ApiError. The backend's message is written to be shown as-is. */
export function ErrorState({ error, onRetry, className }) {
  if (!error) return null
  return (
    <Alert
      tone="error"
      title="Something went wrong"
      className={className}
      action={
        onRetry && (
          <Button size="sm" variant="secondary" onClick={onRetry}>
            Retry
          </Button>
        )
      }
    >
      <p className="whitespace-pre-wrap">{error.message}</p>
      {error.requestId && (
        <p className="mt-1 font-mono text-[11px] opacity-70">request id: {error.requestId}</p>
      )}
    </Alert>
  )
}

export function EmptyState({ icon: Icon, title, children, action, className }) {
  return (
    <div className={cx('flex flex-col items-center justify-center px-6 py-14 text-center', className)}>
      {Icon && (
        <span className="mb-3 rounded-full bg-slate-100 p-3 text-slate-400 dark:bg-slate-800 dark:text-slate-500">
          <Icon className="h-6 w-6" />
        </span>
      )}
      <p className="text-sm font-medium text-slate-700 dark:text-slate-300">{title}</p>
      {children && (
        <div className="mt-1.5 max-w-md text-[13px] text-slate-500 dark:text-slate-400">{children}</div>
      )}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function StatTile({ label, value, sub, tone, icon: Icon }) {
  return (
    <Card className="px-4 py-3.5">
      <div className="flex items-start justify-between gap-2">
        <p className="text-xs font-medium text-slate-500 dark:text-slate-400">{label}</p>
        {Icon && <Icon className="h-4 w-4 shrink-0 text-slate-400" />}
      </div>
      <p className={cx('nums mt-1.5 text-2xl font-semibold tracking-tight', tone || 'text-slate-900 dark:text-slate-100')}>
        {value}
      </p>
      {sub && <p className="mt-0.5 truncate text-[11px] text-slate-500 dark:text-slate-400">{sub}</p>}
    </Card>
  )
}

export function ProgressBar({ value, tone = 'bg-indigo-500', className }) {
  const pct = Math.max(0, Math.min(100, (value || 0) * 100))
  return (
    <div className={cx('h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800', className)}>
      <div className={cx('h-full rounded-full transition-all duration-500', tone)} style={{ width: `${pct}%` }} />
    </div>
  )
}

export function Toggle({ checked, onChange, label, description, disabled }) {
  return (
    <label className={cx('flex items-start gap-3', disabled ? 'opacity-50' : 'cursor-pointer')}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        disabled={disabled}
        onClick={() => !disabled && onChange(!checked)}
        className={cx(
          'relative mt-0.5 inline-flex h-5 w-9 shrink-0 rounded-full transition-colors',
          checked ? 'bg-indigo-600' : 'bg-slate-300 dark:bg-slate-700',
          disabled && 'cursor-not-allowed',
        )}
      >
        <span
          className={cx(
            'absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform',
            checked ? 'translate-x-4.5' : 'translate-x-0.5',
          )}
        />
      </button>
      <span className="min-w-0">
        <span className="block text-sm text-slate-700 dark:text-slate-300">{label}</span>
        {description && (
          <span className="block text-[11px] text-slate-500 dark:text-slate-400">{description}</span>
        )}
      </span>
    </label>
  )
}

export function Select({ label, value, onChange, options, className, ...rest }) {
  return (
    <div className={className}>
      {label && <label className="label">{label}</label>}
      <select className="input" value={value} onChange={(e) => onChange(e.target.value)} {...rest}>
        {options.map((option) => (
          <option key={option.value} value={option.value} disabled={option.disabled}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  )
}

export function Slider({ label, value, onChange, min, max, step = 1, format, className }) {
  return (
    <div className={className}>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="text-xs font-medium text-slate-600 dark:text-slate-400">{label}</span>
        <span className="nums text-xs text-slate-500">{format ? format(value) : value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slate-200 accent-indigo-600 dark:bg-slate-800"
      />
    </div>
  )
}

export function Table({ headers, children, className }) {
  return (
    <div className={cx('overflow-x-auto', className)}>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-200 dark:border-slate-800">
            {headers.map((header) => (
              <th
                key={typeof header === 'string' ? header : header.key}
                className={cx(
                  'px-3 py-2.5 text-left text-[11px] font-semibold uppercase tracking-wide text-slate-500',
                  typeof header !== 'string' && header.align === 'right' && 'text-right',
                )}
              >
                {typeof header === 'string' ? header : header.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-100 dark:divide-slate-800/70">{children}</tbody>
      </table>
    </div>
  )
}

export function Skeleton({ className }) {
  return (
    <div
      className={cx(
        'shimmer relative overflow-hidden rounded bg-slate-200 dark:bg-slate-800',
        className,
      )}
    />
  )
}
