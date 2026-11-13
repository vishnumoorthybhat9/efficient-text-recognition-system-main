import { useEffect, useState } from 'react'
import { Ban, FlaskConical, Play, ScrollText, Terminal } from 'lucide-react'
import { api } from '../api/client.js'
import { usePolling, useAsync } from '../hooks/useAsync.js'
import { PageHeader } from '../components/Layout.jsx'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  ErrorState,
  Loading,
  ProgressBar,
  Table,
} from '../components/ui.jsx'
import { cx, datetime, duration, errorTone, rate, relativeTime } from '../lib/format.js'

const STATUS_TONE = {
  queued: 'slate',
  running: 'indigo',
  completed: 'emerald',
  failed: 'rose',
  cancelled: 'amber',
}

export default function Training() {
  const [starting, setStarting] = useState(null)
  const [error, setError] = useState(null)
  const [logJobId, setLogJobId] = useState(null)
  const [overrides, setOverrides] = useState({})

  const available = useAsync(() => api.availableJobs(), [])

  // Poll while a job is active, then stop: a finished run needs no polling.
  const [active, setActive] = useState(true)
  const jobs = usePolling(() => api.trainingJobs(), { intervalMs: 2500, active })

  useEffect(() => {
    const busy = jobs.data?.busy
    if (busy !== undefined) setActive(Boolean(busy) || jobs.data?.items?.length === 0)
  }, [jobs.data])

  // Keep polling for a little while after a job finishes so the final state
  // and the refreshed model list land without a manual reload.
  useEffect(() => {
    if (!jobs.data?.busy) {
      const handle = setTimeout(() => available.run(), 1500)
      return () => clearTimeout(handle)
    }
    return undefined
  }, [jobs.data?.busy]) // eslint-disable-line react-hooks/exhaustive-deps

  const start = async (key) => {
    setStarting(key)
    setError(null)
    try {
      const patch = overrides[key] || {}
      await api.startTraining({
        job: key,
        epochs: patch.epochs ? Number(patch.epochs) : null,
        batch_size: patch.batch_size ? Number(patch.batch_size) : null,
        max_samples: patch.max_samples ? Number(patch.max_samples) : null,
      })
      setActive(true)
      await available.run()
    } catch (caught) {
      setError(caught)
    } finally {
      setStarting(null)
    }
  }

  const cancel = async (id) => {
    try {
      await api.cancelTraining(id)
    } catch (caught) {
      setError(caught)
    }
  }

  const items = jobs.data?.items || []
  const busy = jobs.data?.busy

  return (
    <div>
      <PageHeader
        title="Training"
        description="Run the three stages of the experiment. Training never starts on its own, and only one job runs at a time — a second would compete for the same GPU memory."
      />

      {error && <ErrorState error={error} className="mb-6" />}

      {busy && (
        <Alert tone="info" title="A job is running" className="mb-6">
          Progress updates below every few seconds. You can keep using the rest of the application,
          though recognition will be slower while the GPU is busy.
        </Alert>
      )}

      <div className="mb-6 grid gap-4 lg:grid-cols-3">
        {available.loading && <Loading />}
        {available.data?.jobs?.map((job, index) => (
          <Card key={job.key} className="flex flex-col">
            <CardHeader
              title={job.label}
              subtitle={`Stage ${index + 1} · ${job.config}`}
              icon={FlaskConical}
            />
            <div className="flex flex-1 flex-col p-5">
              <p className="flex-1 text-[13px] leading-relaxed text-slate-600 dark:text-slate-400">
                {job.description}
              </p>

              {!job.ready && (
                <Alert tone="warning" className="mt-3">
                  {job.blocked_reason}
                </Alert>
              )}

              <div className="mt-4 grid grid-cols-2 gap-2">
                <div>
                  <label className="label">Epochs</label>
                  <input
                    className="input"
                    type="number"
                    min="1"
                    placeholder="config"
                    value={overrides[job.key]?.epochs || ''}
                    onChange={(e) =>
                      setOverrides((o) => ({
                        ...o,
                        [job.key]: { ...o[job.key], epochs: e.target.value },
                      }))
                    }
                  />
                </div>
                <div>
                  <label className="label">Batch size</label>
                  <input
                    className="input"
                    type="number"
                    min="1"
                    placeholder="config"
                    value={overrides[job.key]?.batch_size || ''}
                    onChange={(e) =>
                      setOverrides((o) => ({
                        ...o,
                        [job.key]: { ...o[job.key], batch_size: e.target.value },
                      }))
                    }
                  />
                </div>
              </div>

              <Button
                className="mt-4 w-full"
                icon={Play}
                onClick={() => start(job.key)}
                loading={starting === job.key}
                disabled={!job.ready || busy}
              >
                {busy ? 'A job is already running' : 'Start'}
              </Button>
            </div>
          </Card>
        ))}
      </div>

      <Card>
        <CardHeader
          title="Jobs"
          subtitle={jobs.data ? `${items.length} job(s)` : undefined}
          icon={ScrollText}
        />
        {!jobs.data ? (
          <Loading />
        ) : items.length === 0 ? (
          <EmptyState icon={FlaskConical} title="No training job has been started">
            Start one above, or run the scripts directly:
            <code className="mt-2 block font-mono text-[11px]">
              python scripts/pretrain_simclr.py --config configs/experiments/simclr_pretrain.yaml
            </code>
          </EmptyState>
        ) : (
          <Table headers={['Job', 'Status', 'Progress', 'Latest metrics', 'Started', '']}>
            {items.map((job) => (
              <tr key={job.id} className="align-top">
                <td className="px-3 py-3">
                  <p className="text-sm font-medium">{job.run_name || job.kind}</p>
                  <p className="text-[11px] text-slate-500">{job.config_name}</p>
                </td>
                <td className="px-3 py-3">
                  <Badge tone={STATUS_TONE[job.status] || 'slate'}>{job.status}</Badge>
                  {job.error && (
                    <p className="mt-1 max-w-xs truncate text-[11px] text-rose-600" title={job.error}>
                      {job.error.split('\n')[0]}
                    </p>
                  )}
                </td>
                <td className="w-44 px-3 py-3">
                  <ProgressBar
                    value={job.progress}
                    tone={job.status === 'failed' ? 'bg-rose-500' : 'bg-indigo-500'}
                  />
                  <p className="nums mt-1 text-[11px] text-slate-500">
                    epoch {job.current_epoch}/{job.epochs_total}
                  </p>
                </td>
                <td className="px-3 py-3">
                  {Object.keys(job.latest_metrics || {}).length === 0 ? (
                    <span className="text-xs text-slate-400">—</span>
                  ) : (
                    <div className="nums flex flex-wrap gap-x-3 gap-y-0.5 text-[11px]">
                      {Object.entries(job.latest_metrics)
                        .slice(0, 4)
                        .map(([key, value]) => (
                          <span key={key} className="text-slate-500">
                            {key}{' '}
                            <span
                              className={cx(
                                'font-medium',
                                key.includes('cer') || key.includes('wer')
                                  ? errorTone(value)
                                  : 'text-slate-700 dark:text-slate-300',
                              )}
                            >
                              {typeof value === 'number' ? rate(value) : String(value)}
                            </span>
                          </span>
                        ))}
                    </div>
                  )}
                </td>
                <td className="px-3 py-3 text-[11px] text-slate-500" title={datetime(job.started_at)}>
                  {job.started_at ? relativeTime(job.started_at) : '—'}
                  {job.finished_at && job.started_at && (
                    <span className="block">
                      took{' '}
                      {duration(
                        (new Date(job.finished_at) - new Date(job.started_at)) / 1000,
                      )}
                    </span>
                  )}
                </td>
                <td className="px-3 py-3 text-right">
                  <div className="flex justify-end gap-1">
                    <Button
                      size="sm"
                      variant="ghost"
                      icon={Terminal}
                      onClick={() => setLogJobId(logJobId === job.id ? null : job.id)}
                    >
                      Log
                    </Button>
                    {['queued', 'running'].includes(job.status) && (
                      <Button size="sm" variant="ghost" icon={Ban} onClick={() => cancel(job.id)}>
                        Cancel
                      </Button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </Table>
        )}
      </Card>

      {logJobId && <JobLog jobId={logJobId} onClose={() => setLogJobId(null)} />}
    </div>
  )
}

function JobLog({ jobId, onClose }) {
  const { data } = usePolling(() => api.trainingJobLog(jobId, 300), {
    intervalMs: 3000,
    active: true,
  })

  return (
    <Card className="mt-6">
      <CardHeader
        title={`Log — job ${jobId}`}
        subtitle={data?.path}
        icon={Terminal}
        action={
          <Button size="sm" variant="ghost" onClick={onClose}>
            Close
          </Button>
        }
      />
      <pre className="max-h-96 overflow-auto bg-slate-950 px-5 py-4 font-mono text-[11px] leading-relaxed text-slate-300">
        {data?.log || 'Waiting for output…'}
      </pre>
    </Card>
  )
}
