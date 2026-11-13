/** Formatting helpers shared across pages. */

/** Format a rate in [0,1] as a percentage. */
export function percent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return `${(value * 100).toFixed(digits)}%`
}

/** Format an error rate (CER/WER) to 4 decimals, which is the convention in the literature. */
export function rate(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  return Number(value).toFixed(digits)
}

/** Format milliseconds, switching to seconds once it stops being readable. */
export function ms(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'
  if (value >= 10000) return `${(value / 1000).toFixed(1)} s`
  if (value >= 1000) return `${(value / 1000).toFixed(2)} s`
  if (value >= 10) return `${value.toFixed(0)} ms`
  return `${value.toFixed(2)} ms`
}

/** Format a duration given in seconds. */
export function duration(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(seconds)) return '—'
  if (seconds < 60) return `${seconds.toFixed(1)} s`
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  if (minutes < 60) return `${minutes}m ${rest}s`
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`
}

/** Format a parameter count. */
export function params(count) {
  if (!count) return '—'
  if (count >= 1e9) return `${(count / 1e9).toFixed(2)}B`
  if (count >= 1e6) return `${(count / 1e6).toFixed(2)}M`
  if (count >= 1e3) return `${(count / 1e3).toFixed(1)}K`
  return String(count)
}

/** Format a byte count. */
export function bytes(value) {
  if (!value) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let size = value
  let unit = 0
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024
    unit += 1
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`
}

/** Format an integer with thousands separators. */
export function count(value) {
  if (value === null || value === undefined) return '—'
  return Number(value).toLocaleString()
}

/** Format an ISO timestamp as a readable local date and time. */
export function datetime(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/** Format a timestamp as "3 minutes ago". */
export function relativeTime(value) {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '—'

  const seconds = Math.round((Date.now() - date.getTime()) / 1000)
  if (seconds < 45) return 'just now'

  const steps = [
    [60, 'second'],
    [60, 'minute'],
    [24, 'hour'],
    [7, 'day'],
    [4.345, 'week'],
    [12, 'month'],
  ]

  let value_ = seconds
  let unit = 'second'
  for (const [size, name] of steps) {
    if (value_ < size) break
    value_ /= size
    unit = name
  }
  const rounded = Math.round(value_)
  return `${rounded} ${unit}${rounded === 1 ? '' : 's'} ago`
}

/** Colour classes for a confidence score, so low confidence is visible at a glance. */
export function confidenceTone(value) {
  if (value === null || value === undefined) return 'text-slate-500'
  if (value >= 0.85) return 'text-emerald-600 dark:text-emerald-400'
  if (value >= 0.6) return 'text-amber-600 dark:text-amber-400'
  return 'text-rose-600 dark:text-rose-400'
}

/** Colour classes for an error rate (lower is better). */
export function errorTone(value) {
  if (value === null || value === undefined) return 'text-slate-500'
  if (value <= 0.05) return 'text-emerald-600 dark:text-emerald-400'
  if (value <= 0.2) return 'text-amber-600 dark:text-amber-400'
  return 'text-rose-600 dark:text-rose-400'
}

/** Human label for a model's provenance. */
export function provenanceLabel(provenance) {
  return provenance === 'trained_in_project'
    ? 'Trained in this project'
    : 'Pretrained (third party)'
}

/** Join class names, dropping falsy entries. */
export function cx(...values) {
  return values.filter(Boolean).join(' ')
}
