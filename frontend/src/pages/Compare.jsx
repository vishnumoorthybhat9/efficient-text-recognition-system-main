import { useEffect, useRef, useState } from 'react'
import { AlertTriangle, Images, Layers, Play, Trophy, Zap } from 'lucide-react'
import { api, demoImageAsFile } from '../api/client.js'
import { useAsync } from '../hooks/useAsync.js'
import { PageHeader } from '../components/Layout.jsx'
import { FileDrop } from '../components/FileDrop.jsx'
import { DEFAULT_RECOGNIZE_OPTIONS } from '../components/PreprocessControls.jsx'
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
} from '../components/ui.jsx'
import { cx, errorTone, ms, percent, rate } from '../lib/format.js'

const MODEL_COLUMNS = [
  {
    key: 'cnn',
    family: 'crnn_attention',
    title: 'CNN only',
    description:
      'The available CNN recognizer is CRNN: CNN + BiLSTM + self-attention. It is not a bare CNN.',
  },
  {
    key: 'hybrid',
    family: 'cnn_transformer',
    title: 'CNN + Transformer',
    description: 'CNN visual encoder + Transformer sequence encoder + CTC.',
  },
  {
    key: 'mamba',
    family: 'mamba_ocr',
    title: 'Mamba',
    description: 'CNN visual encoder + bidirectional Mamba sequence layers + CTC.',
  },
]

/**
 * Run one image through the three OCR architecture families with identical
 * preprocessing and render one fixed result column for each family.
 */
export default function Compare() {
  const [file, setFile] = useState(null)
  const [selectedByColumn, setSelectedByColumn] = useState({})
  const [groundTruth, setGroundTruth] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const abort = useRef(null)

  const models = useAsync(() => api.models(), [])
  const demos = useAsync(() => api.demoImages(), [])

  useEffect(() => {
    const available = models.data?.models
    if (!available) return

    setSelectedByColumn((current) => {
      const next = {}
      for (const column of MODEL_COLUMNS) {
        const familyModels = available.filter(
          (model) => model.family === column.family && model.available,
        )
        next[column.key] = familyModels.some((model) => model.id === current[column.key])
          ? current[column.key]
          : familyModels[0]?.id || ''
      }
      return next
    })
  }, [models.data])

  const selectedIds = MODEL_COLUMNS.map((column) => selectedByColumn[column.key]).filter(Boolean)
  const modelsById = new Map((models.data?.models || []).map((model) => [model.id, model]))
  const resultsById = new Map((result?.results || []).map((item) => [item.model_id, item]))

  const run = async () => {
    if (!file || selectedIds.length === 0) return
    abort.current?.abort()
    abort.current = new AbortController()
    setBusy(true)
    setError(null)
    try {
      const data = await api.compare({
        file,
        modelIds: selectedIds,
        options: DEFAULT_RECOGNIZE_OPTIONS,
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
      setFile(await demoImageAsFile(filename))
      const demo = demos.data?.images?.find((d) => d.file === filename)
      setGroundTruth(demo?.text || '')
      setResult(null)
    } catch (caught) {
      setError(caught)
    }
  }

  return (
    <div>
      <PageHeader
        title="Compare models"
        description="Run one image through CNN-based, CNN-Transformer, and Mamba OCR using the same preprocessing, then compare their transcriptions side by side."
      />

      <div className="grid gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
        <div className="space-y-6">
          <Card>
            <CardHeader title="Input image" icon={Images} />
            <div className="p-5">
              <FileDrop file={file} onChange={(f) => { setFile(f); setResult(null) }} disabled={busy} />
              {demos.data?.images?.length > 0 && (
                <div className="mt-4 flex flex-wrap gap-2">
                  {demos.data.images.slice(0, 6).map((demo) => (
                    <button
                      key={demo.file}
                      onClick={() => useDemo(demo.file)}
                      disabled={busy}
                      className="overflow-hidden rounded-lg border border-slate-200 transition-colors hover:border-indigo-400 disabled:opacity-50 dark:border-slate-700"
                      title={demo.text}
                    >
                      <img
                        src={api.demoImageUrl(demo.file)}
                        alt={demo.text}
                        className="h-10 w-auto max-w-[140px] object-contain"
                      />
                    </button>
                  ))}
                </div>
              )}
              <div className="mt-5">
                <label className="label" htmlFor="gt">
                  Ground truth (optional)
                </label>
                <input
                  id="gt"
                  className="input"
                  placeholder="Enables CER and WER for every model"
                  value={groundTruth}
                  onChange={(e) => setGroundTruth(e.target.value)}
                  disabled={busy}
                />
              </div>
            </div>
          </Card>

          <Card>
            <CardHeader title="Three model columns" subtitle="Choose one model per architecture" icon={Layers} />
            {models.loading ? (
              <Loading />
            ) : (
              <div className="space-y-4 p-5">
                {MODEL_COLUMNS.map((column) => {
                  const familyModels = (models.data?.models || []).filter(
                    (model) => model.family === column.family && model.available,
                  )
                  const selectedModel = modelsById.get(selectedByColumn[column.key])
                  return (
                    <div key={column.key}>
                      <label className="label" htmlFor={`model-${column.key}`}>
                        {column.title}
                      </label>
                      <select
                        id={`model-${column.key}`}
                        className="input"
                        value={selectedByColumn[column.key] || ''}
                        disabled={busy || familyModels.length === 0}
                        onChange={(event) =>
                          setSelectedByColumn((current) => ({
                            ...current,
                            [column.key]: event.target.value,
                          }))
                        }
                      >
                        {familyModels.length === 0 ? (
                          <option value="">
                            {column.key === 'hybrid'
                              ? 'Not trained yet'
                              : 'No available checkpoint'}
                          </option>
                        ) : (
                          familyModels.map((model) => (
                            <option key={model.id} value={model.id}>
                              {model.name}
                            </option>
                          ))
                        )}
                      </select>
                      <p className="mt-1 text-[11px] leading-relaxed text-slate-500">
                        {selectedModel?.architecture || column.description}
                      </p>
                    </div>
                  )
                })}
              </div>
            )}
            <div className="border-t border-slate-200 p-4 dark:border-slate-800">
              <Button
                className="w-full"
                icon={Play}
                onClick={run}
                loading={busy}
                disabled={!file || selectedIds.length === 0}
              >
                Compare {selectedIds.length} model{selectedIds.length === 1 ? '' : 's'}
              </Button>
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          {error && <ErrorState error={error} onRetry={run} />}

          {busy && (
            <Card>
              <Loading label="Running every model…" />
            </Card>
          )}

          {!result && !busy && !error && (
            <Card>
              <EmptyState icon={Layers} title="No comparison yet">
                Pick an image and at least one model, then run the comparison. A model that fails
                — Transformer OCR with no network, for instance — is reported on its own and the
                others still return.
              </EmptyState>
            </Card>
          )}

          {result && (
            <>
              <div className="grid gap-4 sm:grid-cols-2">
                <Card className="px-4 py-3.5">
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <Zap className="h-3.5 w-3.5" /> Fastest
                  </div>
                  <p className="mt-1.5 truncate text-lg font-semibold">
                    {result.results.find((r) => r.model_id === result.fastest_model_id)?.model_name ||
                      '—'}
                  </p>
                </Card>
                <Card className="px-4 py-3.5">
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <Trophy className="h-3.5 w-3.5" /> Most accurate
                  </div>
                  <p className="mt-1.5 truncate text-lg font-semibold">
                    {result.most_accurate_model_id
                      ? result.results.find((r) => r.model_id === result.most_accurate_model_id)
                          ?.model_name
                      : 'Add a ground truth'}
                  </p>
                </Card>
              </div>

              {result.ground_truth && (
                <Card>
                  <CardHeader title="Ground truth" />
                  <p className="px-5 py-4 font-mono text-sm">{result.ground_truth}</p>
                </Card>
              )}

              <div className="grid gap-4 lg:grid-cols-3">
                {MODEL_COLUMNS.map((column) => {
                  const modelId = selectedByColumn[column.key]
                  const item = modelId ? resultsById.get(modelId) : null
                  const model = modelId ? modelsById.get(modelId) : null
                  return (
                    <Card key={column.key} className="min-w-0">
                      <CardHeader
                        title={column.title}
                        subtitle={
                          item
                            ? item.error
                              ? 'This model could not run'
                              : `${ms(item.total_ms)} · ${item.line_count} line${item.line_count === 1 ? '' : 's'} · ${percent(item.confidence)} confidence`
                            : model?.name || 'Checkpoint not available'
                        }
                        action={
                          item && !item.error ? (
                            <ProvenanceBadge provenance={item.provenance} />
                          ) : null
                        }
                      />
                      {item?.error ? (
                        <div className="p-4">
                          <Alert tone="warning" title="Not available">
                            <p className="whitespace-pre-wrap">{item.error}</p>
                          </Alert>
                        </div>
                      ) : item ? (
                        <>
                          <pre className="max-h-72 min-h-32 overflow-auto whitespace-pre-wrap break-words px-4 py-4 font-mono text-sm">
                            {item.text || (
                              <span className="text-slate-400">(no text recognised)</span>
                            )}
                          </pre>
                          {item.cer !== null && item.cer !== undefined && (
                            <div className="flex flex-wrap gap-x-4 gap-y-2 border-t border-slate-200 px-4 py-3 text-xs dark:border-slate-800">
                              <span>
                                CER{' '}
                                <span className={cx('nums font-semibold', errorTone(item.cer))}>
                                  {rate(item.cer)}
                                </span>
                              </span>
                              <span>
                                WER{' '}
                                <span className={cx('nums font-semibold', errorTone(item.wer))}>
                                  {rate(item.wer)}
                                </span>
                              </span>
                              <span>
                                Exact{' '}
                                <Badge tone={item.exact_match ? 'emerald' : 'slate'}>
                                  {item.exact_match ? 'yes' : 'no'}
                                </Badge>
                              </span>
                            </div>
                          )}
                        </>
                      ) : (
                        <div className="px-4 py-5 text-sm text-slate-500">
                          {model
                            ? 'Run the comparison to see this model’s transcription.'
                            : column.key === 'hybrid'
                              ? 'Train the CNN + Transformer model to enable this column.'
                              : 'No trained checkpoint for this architecture is currently available.'}
                        </div>
                      )}
                    </Card>
                  )
                })}
              </div>

              {result.results.some((r) => r.error) && (
                <Alert tone="info" icon={AlertTriangle}>
                  One or more models could not run. The comparison above shows the models that
                  did; see each card for the reason.
                </Alert>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
