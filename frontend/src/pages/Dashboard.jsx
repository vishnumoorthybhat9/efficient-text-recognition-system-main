import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Activity,
  ArrowRight,
  BarChart3,
  Clock,
  Cpu,
  Database,
  FileText,
  FlaskConical,
  Layers,
  ScanText,
  Sparkles,
  TrendingDown,
} from 'lucide-react'
import { api } from '../api/client.js'
import { useAsync, usePolling } from '../hooks/useAsync.js'
import { PageHeader } from '../components/Layout.jsx'
import { ChartPanel } from '../components/PlotlyChart.jsx'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardHeader,
  ErrorState,
  Loading,
  ProvenanceBadge,
  StatTile,
} from '../components/ui.jsx'
import { bytes, count, errorTone, ms, percent, rate } from '../lib/format.js'

export default function Dashboard() {
  const health = useAsync(() => api.health(), [])
  const system = useAsync(() => api.systemInfo(), [])
  const models = useAsync(() => api.models(), [])
  const stats = useAsync(() => api.historyStats(), [])
  const comparison = useAsync(() => api.comparison({ split: 'test' }), [])
  const chart = useAsync(() => api.chart('model_comparison', { split: 'test' }), [])
  const [pollTraining, setPollTraining] = useState(true)
  const training = usePolling(() => api.trainingJobs(), {
    intervalMs: 2500,
    active: pollTraining,
  })

  useEffect(() => {
    if (training.data?.busy !== undefined) {
      setPollTraining(Boolean(training.data.busy))
    }
  }, [training.data?.busy])

  const gpu = system.data?.gpus?.[0]
  const trained = models.data?.models?.filter((m) => m.provenance === 'trained_in_project') || []
  const activeTrainingJob = training.data?.items?.find(
    (job) => job.id === training.data?.active_job_id,
  )

  return (
    <div>
      <PageHeader
        title="Dashboard"
        description="Efficient Text Recognition from Images Using Deep Learning and Self-Supervised Learning."
        action={
          <Link to="/recognize">
            <Button icon={ScanText}>Recognise an image</Button>
          </Link>
        }
      />

      {health.error && <ErrorState error={health.error} onRetry={health.run} className="mb-6" />}

      {health.data && !health.data.dataset_ready && (
        <Alert tone="warning" title="No dataset is prepared" className="mb-6">
          Generate the synthetic corpus with{' '}
          <code className="font-mono text-xs">python scripts/make_synthetic_data.py</code>, or
          import a real one from the <Link to="/datasets" className="underline">Datasets</Link> page.
        </Alert>
      )}

      {/* --- headline result --- */}
      {comparison.data?.comparable && (
        <Card className="mb-6 overflow-hidden border-indigo-200 dark:border-indigo-500/30">
          <div className="border-b border-indigo-200 bg-indigo-50 px-5 py-3 dark:border-indigo-500/30 dark:bg-indigo-500/10">
            <div className="flex items-center gap-2">
              <Sparkles className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
              <h2 className="text-sm font-semibold text-indigo-900 dark:text-indigo-200">
                Research result: does self-supervised pretraining help?
              </h2>
            </div>
          </div>
          <div className="grid gap-5 p-5 md:grid-cols-[auto_minmax(0,1fr)]">
            <div className="flex gap-8">
              <div>
                <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
                  From scratch
                </p>
                <p className="nums mt-1 text-2xl font-semibold" style={{ color: 'var(--color-arm-control)' }}>
                  {rate(comparison.data.control?.cer)}
                </p>
                <p className="text-[11px] text-slate-500">CER</p>
              </div>
              <div className="flex items-center">
                <TrendingDown className="h-5 w-5 text-emerald-500" />
              </div>
              <div>
                <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
                  SimCLR pretrained
                </p>
                <p className="nums mt-1 text-2xl font-semibold" style={{ color: 'var(--color-arm-treatment)' }}>
                  {rate(comparison.data.treatment?.cer)}
                </p>
                <p className="text-[11px] text-slate-500">CER</p>
              </div>
              <div className="border-l border-slate-200 pl-8 dark:border-slate-800">
                <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
                  Relative reduction
                </p>
                <p className="nums mt-1 text-2xl font-semibold text-emerald-600 dark:text-emerald-400">
                  {percent(comparison.data.cer_improvement_relative)}
                </p>
                <p className="text-[11px] text-slate-500">
                  on {count(comparison.data.control?.num_samples)} held-out samples
                </p>
              </div>
            </div>
            <p className="self-center text-[13px] leading-relaxed text-slate-600 dark:text-slate-400">
              {comparison.data.summary}
            </p>
          </div>
        </Card>
      )}

      {comparison.data && !comparison.data.comparable && (
        <Alert
          tone="info"
          title={
            training.data?.busy
              ? 'Self-supervised training is in progress'
              : 'The self-supervised comparison is not ready yet'
          }
          className="mb-6"
        >
          <p>
            {training.data?.busy
              ? `Running ${activeTrainingJob?.run_name || 'a training stage'} (${activeTrainingJob?.epochs_completed || 0}/${activeTrainingJob?.epochs_total || 0} epochs). The comparison appears after pretraining, both experiment arms are trained, and both are evaluated on the same test split.`
              : comparison.data.summary}
          </p>
          <Link to="/training" className="mt-2 inline-flex items-center gap-1 text-xs font-medium underline">
            {training.data?.busy ? 'View training progress' : 'Go to Training'}
            <ArrowRight className="h-3 w-3" />
          </Link>
        </Alert>
      )}

      {/* --- tiles --- */}
      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile
          label="Compute device"
          value={health.data?.device?.toUpperCase() || '—'}
          sub={gpu ? `${gpu.name} · ${bytes(gpu.total_memory_mb * 1024 * 1024)}` : system.data?.cpu_name}
          icon={Cpu}
        />
        <StatTile
          label="Models available"
          value={health.data?.models_available ?? '—'}
          sub={`${trained.length} trained in this project`}
          icon={Layers}
        />
        <StatTile
          label="Recognitions run"
          value={count(stats.data?.total_requests)}
          sub={`${count(stats.data?.total_characters)} characters extracted`}
          icon={FileText}
        />
        <StatTile
          label="Mean processing time"
          value={ms(stats.data?.average_total_ms)}
          sub={
            stats.data?.average_confidence
              ? `${percent(stats.data.average_confidence)} mean confidence`
              : 'no requests yet'
          }
          icon={Clock}
        />
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
        <Card>
          <CardHeader
            title="Recognition accuracy by model"
            subtitle="Latest evaluation of each model on the held-out test split"
            icon={BarChart3}
            action={
              <Link to="/evaluation">
                <Button size="sm" variant="secondary">
                  Details
                </Button>
              </Link>
            }
          />
          {chart.loading ? <Loading /> : <ChartPanel chart={chart.data} height={360} />}
        </Card>

        <div className="space-y-6">
          <Card>
            <CardHeader title="Models" icon={Layers} />
            {models.loading ? (
              <Loading />
            ) : (
              <div className="divide-y divide-slate-100 dark:divide-slate-800">
                {models.data?.models?.map((model) => (
                  <div key={model.id} className="px-5 py-3">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium">{model.name}</p>
                        <p className="mt-0.5 text-[11px] text-slate-500">
                          {model.architecture || model.family}
                        </p>
                      </div>
                      <ProvenanceBadge provenance={model.provenance} />
                    </div>
                    <div className="mt-2 flex flex-wrap items-center gap-2">
                      {!model.available && <Badge tone="rose">Unavailable</Badge>}
                      {model.loaded && <Badge tone="indigo">In memory</Badge>}
                      {model.training_summary?.val_cer != null && (
                        <span className={`nums text-[11px] ${errorTone(model.training_summary.val_cer)}`}>
                          val CER {rate(model.training_summary.val_cer)}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Card>

          <Card>
            <CardHeader title="Quick actions" icon={Activity} />
            <div className="grid grid-cols-2 gap-2 p-4">
              {[
                ['/recognize', ScanText, 'Recognise'],
                ['/compare', Layers, 'Compare'],
                ['/evaluation', BarChart3, 'Evaluate'],
                ['/training', FlaskConical, 'Train'],
                ['/datasets', Database, 'Datasets'],
                ['/history', Clock, 'History'],
              ].map(([to, Icon, label]) => (
                <Link
                  key={to}
                  to={to}
                  className="flex flex-col items-center gap-1.5 rounded-lg border border-slate-200 px-3 py-3 text-xs text-slate-600 transition-colors hover:border-indigo-400 hover:text-indigo-700 dark:border-slate-800 dark:text-slate-400 dark:hover:border-indigo-500"
                >
                  <Icon className="h-4 w-4" />
                  {label}
                </Link>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
