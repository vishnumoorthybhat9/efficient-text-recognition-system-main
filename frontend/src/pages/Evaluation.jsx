import { useEffect, useState } from 'react'
import { BarChart3, Play, Sparkles, Table2, TrendingDown } from 'lucide-react'
import { api } from '../api/client.js'
import { useAsync } from '../hooks/useAsync.js'
import { PageHeader } from '../components/Layout.jsx'
import { ChartPanel } from '../components/PlotlyChart.jsx'
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
  Table,
} from '../components/ui.jsx'
import { count, cx, datetime, errorTone, ms, params, percent, rate } from '../lib/format.js'

const CHARTS = [
  { id: 'model_comparison', height: 400 },
  { id: 'accuracy_vs_speed', height: 400 },
  { id: 'training_curves', height: 380 },
  { id: 'simclr_progress', height: 380 },
]

export default function Evaluation() {
  const [split, setSplit] = useState('test')
  const [dataset, setDataset] = useState('')
  const [running, setRunning] = useState(false)
  const [runError, setRunError] = useState(null)
  const [maxSamples, setMaxSamples] = useState('')

  const models = useAsync(() => api.models(), [])
  const datasets = useAsync(() => api.datasets(), [])
  const comparison = useAsync(() => api.comparison({ split, dataset: dataset || undefined }), [split, dataset])
  const runs = useAsync(() => api.evaluationRuns({ split, dataset: dataset || undefined, limit: 50 }), [split, dataset])

  const [charts, setCharts] = useState({})
  useEffect(() => {
    let cancelled = false
    Promise.all(
      CHARTS.map(async ({ id }) => {
        try {
          return [id, await api.chart(id, { split })]
        } catch {
          return [id, null]
        }
      }),
    ).then((entries) => {
      if (!cancelled) setCharts(Object.fromEntries(entries))
    })
    return () => {
      cancelled = true
    }
  }, [split, runs.data])

  const runEvaluation = async () => {
    const available = models.data?.models?.filter((m) => m.available).map((m) => m.id) || []
    if (!available.length) return

    setRunning(true)
    setRunError(null)
    try {
      await api.runEvaluation({
        model_ids: available,
        split,
        dataset: dataset || 'synthetic',
        max_samples: maxSamples ? Number(maxSamples) : null,
        batch_size: 32,
        decode_strategy: 'greedy',
      })
      await Promise.all([runs.run(), comparison.run()])
    } catch (caught) {
      setRunError(caught)
    } finally {
      setRunning(false)
    }
  }

  const prepared = datasets.data?.prepared || []
  const rows = comparison.data?.rows || []

  return (
    <div>
      <PageHeader
        title="Model evaluation"
        description="Accuracy and computational cost on a held-out split. Every number here comes from running the model; nothing is estimated."
        action={
          <div className="flex flex-wrap items-end gap-2">
            <Select
              label="Dataset"
              value={dataset}
              onChange={setDataset}
              options={[
                { value: '', label: 'Any' },
                ...prepared.map((d) => ({ value: d.name, label: d.name })),
              ]}
            />
            <Select
              label="Split"
              value={split}
              onChange={setSplit}
              options={[
                { value: 'test', label: 'test' },
                { value: 'val', label: 'val' },
                { value: 'train', label: 'train' },
              ]}
            />
            <div>
              <label className="label">Max samples</label>
              <input
                className="input w-28"
                type="number"
                min="1"
                placeholder="all"
                value={maxSamples}
                onChange={(e) => setMaxSamples(e.target.value)}
              />
            </div>
            <Button icon={Play} onClick={runEvaluation} loading={running}>
              Evaluate all
            </Button>
          </div>
        }
      />

      {runError && <ErrorState error={runError} className="mb-6" />}
      {running && (
        <Alert tone="info" className="mb-6">
          Running every available model over the {split} split. This takes from a few seconds to a
          couple of minutes depending on the sample count.
        </Alert>
      )}

      {/* --- the headline experiment --- */}
      <Card className="mb-6">
        <CardHeader
          title="Self-supervised pretraining: does it help?"
          subtitle="Both arms share architecture, labeled subset, seed, augmentation and schedule. They differ only in the encoder's initialisation."
          icon={Sparkles}
        />
        {comparison.loading ? (
          <Loading />
        ) : comparison.data?.comparable ? (
          <div className="p-5">
            <div className="mb-4 grid gap-4 sm:grid-cols-3">
              <ArmCard
                title="Control — from scratch"
                row={comparison.data.control}
                colour="var(--color-arm-control)"
              />
              <ArmCard
                title="Treatment — SimCLR pretrained"
                row={comparison.data.treatment}
                colour="var(--color-arm-treatment)"
              />
              <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4 dark:border-emerald-500/30 dark:bg-emerald-500/10">
                <div className="flex items-center gap-2 text-xs font-medium text-emerald-700 dark:text-emerald-300">
                  <TrendingDown className="h-3.5 w-3.5" /> Improvement
                </div>
                <p className="nums mt-2 text-2xl font-semibold text-emerald-700 dark:text-emerald-300">
                  {percent(comparison.data.cer_improvement_relative)}
                </p>
                <p className="mt-0.5 text-[11px] text-emerald-700/80 dark:text-emerald-300/80">
                  relative CER reduction
                </p>
                <p className="nums mt-2 text-[11px] text-emerald-700/80 dark:text-emerald-300/80">
                  {rate(comparison.data.cer_improvement_absolute)} absolute ·{' '}
                  {percent(comparison.data.accuracy_improvement_absolute)} more exact matches
                </p>
              </div>
            </div>
            <p className="text-[13px] leading-relaxed text-slate-600 dark:text-slate-400">
              {comparison.data.summary}
            </p>
          </div>
        ) : (
          <div className="p-5">
            <Alert tone="info">{comparison.data?.summary || 'Not enough data yet.'}</Alert>
          </div>
        )}
      </Card>

      {/* --- results table --- */}
      <Card className="mb-6">
        <CardHeader
          title="All models"
          subtitle={`Latest evaluation of each model on the ${split} split`}
          icon={Table2}
        />
        {runs.loading ? (
          <Loading />
        ) : rows.length === 0 ? (
          <EmptyState icon={BarChart3} title="Nothing evaluated yet">
            Use “Evaluate all” above, or run{' '}
            <code className="font-mono text-xs">python scripts/evaluate.py --all</code>.
          </EmptyState>
        ) : (
          <Table
            headers={[
              { key: 'model', label: 'Model' },
              { key: 'cer', label: 'CER', align: 'right' },
              { key: 'cerNorm', label: 'CER (case-ins.)', align: 'right' },
              { key: 'wer', label: 'WER', align: 'right' },
              { key: 'exact', label: 'Exact', align: 'right' },
              { key: 'charAcc', label: 'Char acc.', align: 'right' },
              { key: 'latency', label: 'ms/img', align: 'right' },
              { key: 'throughput', label: 'img/s', align: 'right' },
              { key: 'params', label: 'Params', align: 'right' },
              { key: 'n', label: 'Samples', align: 'right' },
            ]}
          >
            {rows.map((row) => (
              <tr key={row.model_id} className="hover:bg-slate-50 dark:hover:bg-slate-800/40">
                <td className="px-3 py-2.5">
                  <div className="flex items-center gap-2">
                    {row.experiment_arm === 'control' && (
                      <span className="h-2 w-2 rounded-full" style={{ background: 'var(--color-arm-control)' }} />
                    )}
                    {row.experiment_arm === 'treatment' && (
                      <span className="h-2 w-2 rounded-full" style={{ background: 'var(--color-arm-treatment)' }} />
                    )}
                    <span className="font-medium">{row.model_name}</span>
                    <ProvenanceBadge provenance={row.provenance} />
                  </div>
                  <span className="mt-0.5 block text-[11px] text-slate-500">{row.architecture}</span>
                </td>
                <td className={cx('nums px-3 py-2.5 text-right font-semibold', errorTone(row.cer))}>
                  {rate(row.cer)}
                </td>
                <td className={cx('nums px-3 py-2.5 text-right', errorTone(row.cer_normalized))}>
                  {rate(row.cer_normalized)}
                </td>
                <td className={cx('nums px-3 py-2.5 text-right', errorTone(row.wer))}>{rate(row.wer)}</td>
                <td className="nums px-3 py-2.5 text-right">{percent(row.sequence_accuracy)}</td>
                <td className="nums px-3 py-2.5 text-right">{percent(row.character_accuracy)}</td>
                <td className="nums px-3 py-2.5 text-right">{row.avg_inference_ms?.toFixed(2)}</td>
                <td className="nums px-3 py-2.5 text-right">{row.throughput_images_per_s?.toFixed(0)}</td>
                <td className="nums px-3 py-2.5 text-right">{params(row.parameter_count)}</td>
                <td className="nums px-3 py-2.5 text-right text-slate-500">{count(row.num_samples)}</td>
              </tr>
            ))}
          </Table>
        )}
        {rows.length > 0 && (
          <p className="border-t border-slate-200 px-5 py-3 text-[11px] text-slate-500 dark:border-slate-800">
            CER and WER are micro-averaged (total edits ÷ total reference length). “Exact” is the
            fraction of samples transcribed perfectly. Latency excludes image loading and discards
            warm-up batches. Models marked “Pretrained” were trained by third parties and are shown
            as external baselines only — read their <strong>CER (case-ins.)</strong> column, which
            ignores letter case and punctuation: TrOCR was trained on all-caps receipt text, so the
            strict column charges it for a convention difference rather than for misreading.
          </p>
        )}
      </Card>

      {/* --- charts --- */}
      <div className="grid gap-6 xl:grid-cols-2">
        {CHARTS.map(({ id, height }) => (
          <Card key={id}>
            <CardHeader title={charts[id]?.title || id} icon={BarChart3} />
            {charts[id] === undefined ? (
              <Loading />
            ) : (
              <ChartPanel chart={charts[id]} height={height} />
            )}
          </Card>
        ))}
      </div>

      {/* --- raw runs --- */}
      {runs.data?.items?.length > 0 && (
        <Card className="mt-6">
          <CardHeader title="Evaluation history" subtitle={`${runs.data.total} stored run(s)`} />
          <Table
            headers={['When', 'Model', 'Dataset', 'CER', 'Samples', 'Device']}
          >
            {runs.data.items.map((run) => (
              <tr key={run.id} className="text-slate-600 dark:text-slate-400">
                <td className="px-3 py-2 text-xs">{datetime(run.created_at)}</td>
                <td className="px-3 py-2 text-xs font-medium text-slate-800 dark:text-slate-200">
                  {run.model_name}
                </td>
                <td className="px-3 py-2 text-xs">
                  {run.dataset}/{run.split}
                </td>
                <td className={cx('nums px-3 py-2 text-xs', errorTone(run.cer))}>{rate(run.cer)}</td>
                <td className="nums px-3 py-2 text-xs">{count(run.num_samples)}</td>
                <td className="px-3 py-2 text-xs">
                  <Badge tone={run.device?.startsWith('cuda') ? 'indigo' : 'slate'}>{run.device}</Badge>
                </td>
              </tr>
            ))}
          </Table>
        </Card>
      )}
    </div>
  )
}

function ArmCard({ title, row, colour }) {
  if (!row) return null
  return (
    <div className="rounded-lg border border-slate-200 p-4 dark:border-slate-800">
      <div className="flex items-center gap-2">
        <span className="h-2 w-2 rounded-full" style={{ background: colour }} />
        <p className="text-xs font-medium text-slate-600 dark:text-slate-400">{title}</p>
      </div>
      <p className="nums mt-2 text-2xl font-semibold" style={{ color: colour }}>
        {rate(row.cer)}
      </p>
      <p className="mt-0.5 text-[11px] text-slate-500">CER</p>
      <dl className="nums mt-3 space-y-1 text-[11px] text-slate-500">
        <div className="flex justify-between">
          <dt>WER</dt>
          <dd>{rate(row.wer)}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Exact match</dt>
          <dd>{percent(row.sequence_accuracy)}</dd>
        </div>
        <div className="flex justify-between">
          <dt>Latency</dt>
          <dd>{ms(row.avg_inference_ms)}</dd>
        </div>
      </dl>
    </div>
  )
}
