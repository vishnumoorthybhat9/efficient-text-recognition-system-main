import { useEffect, useMemo, useRef, useState } from 'react'
import { Images, Play, ScanText, Settings2, Target } from 'lucide-react'
import { api, demoImageAsFile } from '../api/client.js'
import { useAsync } from '../hooks/useAsync.js'
import { PageHeader } from '../components/Layout.jsx'
import { FileDrop } from '../components/FileDrop.jsx'
import { ExtractionResult } from '../components/ExtractionResult.jsx'
import { ImageCompare } from '../components/ImageCompare.jsx'
import {
  DEFAULT_RECOGNIZE_OPTIONS,
  PreprocessControls,
  RecognitionControls,
} from '../components/PreprocessControls.jsx'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  ErrorState,
  Loading,
  ProvenanceBadge,
  Select,
} from '../components/ui.jsx'
import { cx, errorTone, ms, percent, rate } from '../lib/format.js'

export default function Recognize() {
  const [file, setFile] = useState(null)
  const [modelId, setModelId] = useState('')
  const [options, setOptions] = useState(DEFAULT_RECOGNIZE_OPTIONS)
  const [groundTruth, setGroundTruth] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [tab, setTab] = useState('recognition')
  const abort = useRef(null)

  const models = useAsync(() => api.models(), [])
  const demos = useAsync(() => api.demoImages(), [])

  useEffect(() => {
    if (models.data?.default_model_id && !modelId) {
      setModelId(models.data.default_model_id)
    }
  }, [models.data, modelId])

  const selected = useMemo(
    () => models.data?.models?.find((m) => m.id === modelId),
    [models.data, modelId],
  )

  const available = models.data?.models?.filter((m) => m.available) || []

  const run = async () => {
    if (!file) return
    abort.current?.abort()
    abort.current = new AbortController()

    setBusy(true)
    setError(null)
    try {
      const data = await api.recognize({
        file,
        modelId,
        options,
        groundTruth: groundTruth.trim() || undefined,
        signal: abort.current.signal,
      })
      setResult(data)
    } catch (caught) {
      if (caught.name !== 'AbortError') setError(caught)
      setResult(null)
    } finally {
      setBusy(false)
    }
  }

  const useDemo = async (filename) => {
    try {
      setError(null)
      const demoFile = await demoImageAsFile(filename)
      setFile(demoFile)
      setResult(null)
      const demo = demos.data?.images?.find((d) => d.file === filename)
      setGroundTruth(demo?.text || '')
    } catch (caught) {
      setError(caught)
    }
  }

  // CER/WER are computed server-side when a ground truth is supplied.
  const scored = result && groundTruth.trim()

  return (
    <div>
      <PageHeader
        title="Recognise text"
        description="Upload a handwritten page to clean the scan, detect its text lines, and transcribe them. You can edit or export the result."
      />

      {models.error && <ErrorState error={models.error} onRetry={models.run} className="mb-6" />}

      {models.data && available.length === 0 && (
        <Alert tone="warning" title="No model is available yet" className="mb-6">
          Train one from the Training page, or run{' '}
          <code className="font-mono text-xs">python scripts/train_crnn.py</code> in a terminal.
        </Alert>
      )}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
        {/* ---------------- left column ---------------- */}
        <div className="space-y-6">
          <Card>
            <CardHeader title="Input image" icon={Images} />
            <div className="p-5">
              <FileDrop file={file} onChange={(f) => { setFile(f); setResult(null) }} disabled={busy} />

              {demos.data?.images?.length > 0 && (
                <div className="mt-4">
                  <p className="mb-2 text-xs font-medium text-slate-500">
                    Or try a bundled sample (no upload needed)
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {demos.data.images.map((demo) => (
                      <button
                        key={demo.file}
                        onClick={() => useDemo(demo.file)}
                        disabled={busy}
                        className="group overflow-hidden rounded-lg border border-slate-200 bg-white transition-colors hover:border-indigo-400 disabled:opacity-50 dark:border-slate-700 dark:bg-slate-950"
                        title={demo.text}
                      >
                        <img
                          src={api.demoImageUrl(demo.file)}
                          alt={demo.text}
                          className="h-11 w-auto max-w-[180px] object-contain"
                        />
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <div className="mt-5">
                <label className="label" htmlFor="ground-truth">
                  Ground truth (optional)
                </label>
                <input
                  id="ground-truth"
                  className="input"
                  placeholder="Type what the image actually says to score this result"
                  value={groundTruth}
                  onChange={(e) => setGroundTruth(e.target.value)}
                  disabled={busy}
                />
                <p className="mt-1 text-[11px] text-slate-500">
                  Supplying this computes CER and WER for this image and stores them with the
                  history entry.
                </p>
              </div>

              <Button
                className="mt-5 w-full"
                size="lg"
                icon={Play}
                onClick={run}
                loading={busy}
                disabled={!file || !modelId || available.length === 0}
              >
                {busy ? 'Recognising…' : 'Recognise text'}
              </Button>
            </div>
          </Card>

          {error && <ErrorState error={error} onRetry={run} />}

          {busy && !result && (
            <Card>
              <Loading label="Running the model…" />
            </Card>
          )}

          {result && (
            <>
              <Card>
                <CardHeader
                  title="Extracted text"
                  subtitle={`${result.model_name} · ${ms(result.timings_ms.total)} on ${result.device}`}
                  icon={ScanText}
                  action={<ProvenanceBadge provenance={selected?.provenance} />}
                />
                <ExtractionResult
                  result={result}
                  modelName={result.model_name}
                  busy={busy}
                  onRerun={run}
                  onClear={() => { setResult(null); setError(null) }}
                />
              </Card>

              <ImageCompare file={file} options={options} />

              {scored && (
                <Card>
                  <CardHeader title="Accuracy against your ground truth" icon={Target} />
                  <div className="grid grid-cols-3 divide-x divide-slate-200 dark:divide-slate-800">
                    <ScoreCell label="CER" value={result.cer} />
                    <ScoreCell label="WER" value={result.wer} />
                    <div className="px-5 py-4 text-center">
                      <p className="text-xs text-slate-500">Exact match</p>
                      <p className="mt-1 text-lg font-semibold">
                        {result.exact_match ? 'Yes' : 'No'}
                      </p>
                    </div>
                  </div>
                </Card>
              )}

              <Card>
                <CardHeader
                  title="How the time was spent"
                  subtitle="Preprocessing and line detection are shared by every model; only inference differs."
                />
                <div className="space-y-2.5 p-5">
                  {[
                    ['Preprocessing', result.timings_ms.preprocess, 'bg-sky-500'],
                    ['Line detection', result.timings_ms.segment, 'bg-violet-500'],
                    ['Model inference', result.timings_ms.inference, 'bg-indigo-500'],
                  ].map(([label, value, tone]) => (
                    <div key={label}>
                      <div className="mb-1 flex justify-between text-xs">
                        <span className="text-slate-600 dark:text-slate-400">{label}</span>
                        <span className="nums text-slate-500">{ms(value)}</span>
                      </div>
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
                        <div
                          className={cx('h-full rounded-full', tone)}
                          style={{
                            width: `${Math.min(100, (value / Math.max(result.timings_ms.total, 1)) * 100)}%`,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                  <div className="flex justify-between border-t border-slate-200 pt-2.5 text-xs font-medium dark:border-slate-800">
                    <span>Total</span>
                    <span className="nums">{ms(result.timings_ms.total)}</span>
                  </div>
                </div>
              </Card>

              <Card>
                <CardHeader
                  title={`Detected lines (${result.line_count})`}
                  subtitle={
                    result.segmentation.used_whole_image
                      ? 'Treated as a single pre-cropped text line'
                      : `Median character height ${result.segmentation.median_char_height?.toFixed(0)} px`
                  }
                />
                {result.lines.length === 0 ? (
                  <EmptyState title="No lines were detected" />
                ) : (
                  <div className="divide-y divide-slate-100 dark:divide-slate-800">
                    {result.lines.map((line) => (
                      <div key={line.index} className="flex items-center gap-3 px-5 py-2.5 text-sm">
                        <span className="nums w-6 shrink-0 text-right text-[11px] text-slate-400">
                          {line.index + 1}
                        </span>
                        <span className="min-w-0 flex-1 truncate font-mono">{line.text}</span>
                        {line.region && (
                          <span className="nums shrink-0 font-mono text-[10px] text-slate-400">
                            {line.region.width}×{line.region.height}
                          </span>
                        )}
                        <Badge tone={line.confidence >= 0.8 ? 'emerald' : line.confidence >= 0.6 ? 'amber' : 'rose'}>
                          {percent(line.confidence, 0)}
                        </Badge>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            </>
          )}
        </div>

        {/* ---------------- right column ---------------- */}
        <div className="space-y-6">
          <Card>
            <CardHeader title="Model" icon={Settings2} />
            <div className="space-y-4 p-5">
              {models.loading ? (
                <Loading label="Loading models…" />
              ) : (
                <>
                  <Select
                    value={modelId}
                    onChange={setModelId}
                    disabled={busy}
                    options={
                      available.length
                        ? available.map((m) => ({ value: m.id, label: m.name }))
                        : [{ value: '', label: 'No models available' }]
                    }
                  />
                  {selected && (
                    <div className="rounded-lg bg-slate-50 p-3 dark:bg-slate-950">
                      <div className="mb-1.5 flex items-center gap-2">
                        <ProvenanceBadge provenance={selected.provenance} />
                        {selected.loaded && <Badge tone="indigo">In memory</Badge>}
                      </div>
                      <p className="text-[11px] leading-relaxed text-slate-600 dark:text-slate-400">
                        {selected.description}
                      </p>
                      {selected.training_summary?.val_cer != null && (
                        <p className="nums mt-2 text-[11px] text-slate-500">
                          Validation CER during training:{' '}
                          <span className={errorTone(selected.training_summary.val_cer)}>
                            {rate(selected.training_summary.val_cer)}
                          </span>
                        </p>
                      )}
                    </div>
                  )}
                </>
              )}
            </div>
          </Card>

          <Card>
            <div className="flex border-b border-slate-200 dark:border-slate-800">
              {[
                ['recognition', 'Recognition'],
                ['preprocess', 'Enhancement'],
              ].map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => setTab(key)}
                  className={cx(
                    'flex-1 px-4 py-2.5 text-xs font-medium transition-colors',
                    tab === key
                      ? 'border-b-2 border-indigo-600 text-indigo-700 dark:text-indigo-400'
                      : 'text-slate-500 hover:text-slate-700 dark:hover:text-slate-300',
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="p-5">
              {tab === 'recognition' ? (
                <RecognitionControls
                  value={options}
                  onChange={setOptions}
                  supportsBeam={selected?.supports_beam_search !== false}
                  disabled={busy}
                />
              ) : (
                <PreprocessControls
                  value={options.preprocess}
                  onChange={(preprocess) => setOptions({ ...options, preprocess })}
                  disabled={busy}
                  compact
                />
              )}
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}

function ScoreCell({ label, value }) {
  return (
    <div className="px-5 py-4 text-center">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={cx('nums mt-1 text-lg font-semibold', errorTone(value))}>{rate(value)}</p>
    </div>
  )
}
