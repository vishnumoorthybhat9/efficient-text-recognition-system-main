import { useState } from 'react'
import { Check, Copy, Download, FileText } from 'lucide-react'
import { Button, EmptyState } from './ui.jsx'
import { confidenceTone, cx, percent } from '../lib/format.js'

/**
 * Extracted text with working copy and download controls.
 *
 * Copy falls back to a hidden textarea plus `execCommand`: the async
 * clipboard API is unavailable on an insecure origin, which is exactly how
 * this app is served during a local demo (http://localhost).
 */
export function TextOutput({ text, lines = [], confidence, filename = 'extracted-text', className }) {
  const [copied, setCopied] = useState(false)
  const [showPerLine, setShowPerLine] = useState(false)

  const copy = async () => {
    const payload = text || ''
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(payload)
      } else {
        const area = document.createElement('textarea')
        area.value = payload
        area.style.position = 'fixed'
        area.style.opacity = '0'
        document.body.appendChild(area)
        area.select()
        document.execCommand('copy')
        document.body.removeChild(area)
      }
      setCopied(true)
      setTimeout(() => setCopied(false), 1800)
    } catch {
      setCopied(false)
    }
  }

  const download = (extension) => {
    let payload = text || ''
    let mime = 'text/plain;charset=utf-8'

    if (extension === 'json') {
      payload = JSON.stringify(
        {
          text,
          confidence,
          line_count: lines.length,
          lines: lines.map((line) => ({
            index: line.index,
            text: line.text,
            confidence: line.confidence,
            region: line.region,
          })),
          exported_at: new Date().toISOString(),
        },
        null,
        2,
      )
      mime = 'application/json;charset=utf-8'
    }

    const blob = new Blob([payload], { type: mime })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = `${filename}.${extension}`
    document.body.appendChild(anchor)
    anchor.click()
    document.body.removeChild(anchor)
    // Revoke on the next tick; revoking immediately can cancel the download
    // in some browsers.
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }

  if (!text) {
    return (
      <EmptyState icon={FileText} title="No text was recognised" className={className}>
        The model did not find readable text. Try enabling contrast enhancement or binarisation in
        the preprocessing options, or check that the image contains text.
      </EmptyState>
    )
  }

  return (
    <div className={className}>
      <div className="flex flex-wrap items-center justify-between gap-2 px-5 py-3">
        <div className="flex items-center gap-3 text-xs text-slate-500">
          <span>{text.length} characters</span>
          <span>·</span>
          <span>{text.split(/\s+/).filter(Boolean).length} words</span>
          <span>·</span>
          <span>{lines.length || 1} line{lines.length === 1 ? '' : 's'}</span>
          {confidence !== undefined && confidence !== null && (
            <>
              <span>·</span>
              <span className={confidenceTone(confidence)}>{percent(confidence)} confidence</span>
            </>
          )}
        </div>
        <div className="flex gap-2">
          {lines.length > 1 && (
            <Button size="sm" variant="ghost" onClick={() => setShowPerLine((v) => !v)}>
              {showPerLine ? 'Show as text' : 'Show per line'}
            </Button>
          )}
          <Button size="sm" variant="secondary" icon={copied ? Check : Copy} onClick={copy}>
            {copied ? 'Copied' : 'Copy'}
          </Button>
          <Button size="sm" variant="secondary" icon={Download} onClick={() => download('txt')}>
            .txt
          </Button>
          <Button size="sm" variant="secondary" icon={Download} onClick={() => download('json')}>
            .json
          </Button>
        </div>
      </div>

      <div className="border-t border-slate-200 px-5 py-4 dark:border-slate-800">
        {showPerLine && lines.length > 1 ? (
          <ul className="space-y-1.5">
            {lines.map((line) => (
              <li
                key={line.index}
                className="flex items-start gap-3 rounded-lg bg-slate-50 px-3 py-2 dark:bg-slate-950"
              >
                <span className="nums mt-0.5 w-6 shrink-0 text-right text-[11px] text-slate-400">
                  {line.index + 1}
                </span>
                <span className="min-w-0 flex-1 break-words font-mono text-sm">{line.text}</span>
                <span className={cx('nums shrink-0 text-[11px]', confidenceTone(line.confidence))}>
                  {percent(line.confidence, 0)}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <pre className="max-h-96 overflow-auto whitespace-pre-wrap break-words font-mono text-sm leading-relaxed text-slate-800 dark:text-slate-200">
            {text}
          </pre>
        )}
      </div>
    </div>
  )
}
