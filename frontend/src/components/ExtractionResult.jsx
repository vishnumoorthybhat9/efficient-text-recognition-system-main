import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
  Check,
  Copy,
  Download,
  Eye,
  FileText,
  Pencil,
  RotateCcw,
  Trash2,
  Undo2,
} from 'lucide-react'
import { api, ApiError } from '../api/client.js'
import { Alert, Badge, Button, EmptyState } from './ui.jsx'
import { confidenceTone, cx, ms, percent } from '../lib/format.js'

/**
 * The extraction panel: editable transcription, exports and uncertainty.
 *
 * Two principles drive the design.
 *
 * **The model's output is preserved separately from the user's edits.** The
 * original transcription is what the system actually produced and is what any
 * accuracy figure must be computed against; the edited text is the user's
 * correction. Conflating them would let a hand-fixed result be reported as
 * model accuracy.
 *
 * **Uncertainty is shown, not hidden.** Lines the model was unsure about are
 * marked rather than presented with the same authority as confident ones.
 */
export function ExtractionResult({
  result,
  modelName,
  onRerun,
  onClear,
  busy = false,
  className,
}) {
  const original = result?.text ?? ''
  const [text, setText] = useState(original)
  const [copied, setCopied] = useState(false)
  const [exporting, setExporting] = useState(null)
  const [exportError, setExportError] = useState(null)
  const [view, setView] = useState('text')
  const [threshold, setThreshold] = useState(0.6)
  const areaRef = useRef(null)

  // A new result replaces the editable copy; edits belong to the result that
  // produced them.
  useEffect(() => {
    setText(original)
    setExportError(null)
  }, [original, result?.id])

  const edited = text !== original
  const lines = result?.lines ?? []

  const uncertain = useMemo(
    () => lines.filter((line) => (line.confidence ?? 1) < threshold),
    [lines, threshold],
  )

  if (!result) return null

  if (!original.trim()) {
    return (
      <EmptyState icon={FileText} title="No text was recognised" className={className}>
        The model found no readable text. Try enabling <strong>Enhance contrast</strong> or
        <strong> Binarisation</strong> in the preprocessing options, or check that the image
        actually contains text. The Preprocessing Lab shows which lines were detected.
      </EmptyState>
    )
  }

  const copy = async () => {
    try {
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text)
      } else {
        const area = document.createElement('textarea')
        area.value = text
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
      setExportError(new ApiError('Could not copy to the clipboard.'))
    }
  }

  const download = async (format) => {
    setExporting(format)
    setExportError(null)
    try {
      const blob = await api.exportFile(format, {
        text,
        filename: `extracted-text-${result.id ?? 'result'}`,
        title: 'Extracted text',
        model_name: modelName || result.model_name,
        device: result.device,
        total_ms: result.timings_ms?.total,
        confidence: result.confidence,
        lines: lines.map((line) => ({ text: line.text, confidence: line.confidence })),
        low_confidence_threshold: threshold,
      })

      const url = URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `extracted-text-${result.id ?? 'result'}.${format}`
      document.body.appendChild(anchor)
      anchor.click()
      document.body.removeChild(anchor)
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    } catch (caught) {
      setExportError(caught)
    } finally {
      setExporting(null)
    }
  }

  return (
    <div className={className}>
      {/* --- toolbar --- */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 px-5 py-3 dark:border-slate-800">
        <div className="flex flex-wrap items-center gap-2.5 text-xs text-slate-500">
          <span>{text.length} characters</span>
          <span>·</span>
          <span>{text.split(/\s+/).filter(Boolean).length} words</span>
          <span>·</span>
          <span>{text.split('\n').length} lines</span>
          {result.confidence != null && (
            <>
              <span>·</span>
              <span className={confidenceTone(result.confidence)}>
                {percent(result.confidence)} confidence
              </span>
            </>
          )}
          {result.timings_ms?.total != null && (
            <>
              <span>·</span>
              <span>{ms(result.timings_ms.total)}</span>
            </>
          )}
          {edited && (
            <Badge tone="amber">
              <Pencil className="mr-1 h-3 w-3" />
              edited
            </Badge>
          )}
        </div>

        <div className="flex flex-wrap gap-1.5">
          <Button size="sm" variant="ghost" icon={view === 'text' ? Eye : FileText}
            onClick={() => setView(view === 'text' ? 'lines' : 'text')}>
            {view === 'text' ? 'Per line' : 'As text'}
          </Button>
          <Button size="sm" variant="secondary" icon={copied ? Check : Copy} onClick={copy}>
            {copied ? 'Copied' : 'Copy'}
          </Button>
          <Button
            size="sm" variant="secondary" icon={Download}
            loading={exporting === 'txt'} onClick={() => download('txt')}
          >
            .txt
          </Button>
          <Button
            size="sm" variant="secondary" icon={Download}
            loading={exporting === 'docx'} onClick={() => download('docx')}
          >
            .docx
          </Button>
          {onRerun && (
            <Button size="sm" variant="secondary" icon={RotateCcw} onClick={onRerun} disabled={busy}>
              Run again
            </Button>
          )}
          {onClear && (
            <Button size="sm" variant="ghost" icon={Trash2} onClick={onClear} disabled={busy}>
              Clear
            </Button>
          )}
        </div>
      </div>

      {exportError && (
        <div className="px-5 pt-4">
          <Alert tone="error" title="Export failed">
            {exportError.message}
          </Alert>
        </div>
      )}

      {/* --- uncertainty --- */}
      {lines.length > 0 && (
        <div className="flex flex-wrap items-center gap-3 border-b border-slate-200 bg-slate-50 px-5 py-2.5 dark:border-slate-800 dark:bg-slate-950">
          <span className="flex items-center gap-1.5 text-[11px] font-medium text-slate-600 dark:text-slate-400">
            <AlertTriangle className="h-3.5 w-3.5" />
            Flag lines below
          </span>
          <input
            type="range" min={0} max={0.95} step={0.05} value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="h-1.5 w-32 cursor-pointer appearance-none rounded-full bg-slate-200 accent-amber-500 dark:bg-slate-800"
          />
          <span className="nums text-[11px] text-slate-500">
            {threshold === 0 ? 'off' : percent(threshold, 0)}
          </span>
          {threshold > 0 && (
            <Badge tone={uncertain.length ? 'amber' : 'emerald'}>
              {uncertain.length} of {lines.length} line(s) uncertain
            </Badge>
          )}
          <span className="text-[11px] text-slate-400">
            Confidence is the model's own score, not a guarantee — always check against the image.
          </span>
        </div>
      )}

      {/* --- body --- */}
      {view === 'text' ? (
        <div className="p-5">
          <label className="label" htmlFor="extracted">
            Extracted text — editable
          </label>
          <textarea
            id="extracted"
            ref={areaRef}
            className="input min-h-64 resize-y whitespace-pre font-mono text-sm leading-relaxed"
            spellCheck={false}
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
            <p className="text-[11px] text-slate-500">
              Corrections stay here only. The model's original output is kept separately and is
              what any accuracy figure is measured against.
            </p>
            {edited && (
              <Button size="sm" variant="ghost" icon={Undo2} onClick={() => setText(original)}>
                Revert to original
              </Button>
            )}
          </div>
        </div>
      ) : (
        <div className="divide-y divide-slate-100 p-1 dark:divide-slate-800">
          {lines.map((line) => {
            const low = (line.confidence ?? 1) < threshold
            return (
              <div
                key={line.index}
                className={cx(
                  'flex items-start gap-3 px-4 py-2 text-sm',
                  low && 'bg-amber-50 dark:bg-amber-500/10',
                )}
              >
                <span className="nums mt-0.5 w-6 shrink-0 text-right text-[11px] text-slate-400">
                  {line.index + 1}
                </span>
                <span className="min-w-0 flex-1 break-words font-mono">{line.text}</span>
                {line.region && (
                  <span className="nums shrink-0 font-mono text-[10px] text-slate-400">
                    {line.region.width}×{line.region.height}
                  </span>
                )}
                <Badge tone={low ? 'amber' : line.confidence >= 0.85 ? 'emerald' : 'slate'}>
                  {percent(line.confidence, 0)}
                </Badge>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
