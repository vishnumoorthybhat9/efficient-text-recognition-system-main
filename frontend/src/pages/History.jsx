import { useEffect, useState } from 'react'
import {
  Check,
  Clock,
  FileText,
  Image as ImageIcon,
  Search,
  Target,
  Trash2,
  X,
} from 'lucide-react'
import { api } from '../api/client.js'
import { useAsync } from '../hooks/useAsync.js'
import { PageHeader } from '../components/Layout.jsx'
import {
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  ErrorState,
  Loading,
  Select,
  StatTile,
  Table,
} from '../components/ui.jsx'
import {
  confidenceTone,
  count,
  cx,
  datetime,
  errorTone,
  ms,
  percent,
  rate,
  relativeTime,
} from '../lib/format.js'

const PAGE_SIZE = 25

export default function History() {
  const [page, setPage] = useState(0)
  const [search, setSearch] = useState('')
  const [debounced, setDebounced] = useState('')
  const [modelFilter, setModelFilter] = useState('')
  const [scoredOnly, setScoredOnly] = useState(false)
  const [detailId, setDetailId] = useState(null)

  // Debounce so a search does not fire a request per keystroke.
  useEffect(() => {
    const handle = setTimeout(() => {
      setDebounced(search)
      setPage(0)
    }, 350)
    return () => clearTimeout(handle)
  }, [search])

  const models = useAsync(() => api.models(), [])
  const stats = useAsync(() => api.historyStats(), [])
  const list = useAsync(
    () =>
      api.history({
        limit: PAGE_SIZE,
        offset: page * PAGE_SIZE,
        search: debounced || undefined,
        model_id: modelFilter || undefined,
        scored_only: scoredOnly || undefined,
      }),
    [page, debounced, modelFilter, scoredOnly],
  )

  const refresh = () => {
    list.run()
    stats.run()
  }

  const remove = async (id) => {
    await api.deleteHistoryItem(id)
    if (detailId === id) setDetailId(null)
    refresh()
  }

  const clearAll = async () => {
    if (!window.confirm('Delete every history entry and its stored image? This cannot be undone.')) {
      return
    }
    await api.clearHistory()
    setDetailId(null)
    setPage(0)
    refresh()
  }

  const items = list.data?.items || []
  const total = list.data?.total || 0
  const pages = Math.ceil(total / PAGE_SIZE)

  return (
    <div>
      <PageHeader
        title="Recognition history"
        description="Every recognition is stored in SQLite with its result, timings and settings. Add a ground truth to any entry to score it."
        action={
          total > 0 && (
            <Button variant="danger" size="sm" icon={Trash2} onClick={clearAll}>
              Clear history
            </Button>
          )
        }
      />

      <div className="mb-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile label="Total requests" value={count(stats.data?.total_requests)} icon={FileText} />
        <StatTile
          label="Characters extracted"
          value={count(stats.data?.total_characters)}
          icon={FileText}
        />
        <StatTile
          label="Mean confidence"
          value={percent(stats.data?.average_confidence)}
          sub={`mean ${ms(stats.data?.average_total_ms)} per request`}
          icon={Clock}
        />
        <StatTile
          label="Mean CER"
          value={stats.data?.average_cer != null ? rate(stats.data.average_cer) : '—'}
          sub={
            stats.data?.scored_requests
              ? `over ${count(stats.data.scored_requests)} scored entries`
              : 'add a ground truth to score entries'
          }
          tone={stats.data?.average_cer != null ? errorTone(stats.data.average_cer) : undefined}
          icon={Target}
        />
      </div>

      <Card className="mb-6">
        <div className="flex flex-wrap items-end gap-3 p-4">
          <div className="min-w-56 flex-1">
            <label className="label">Search</label>
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
              <input
                className="input pl-9"
                placeholder="Recognised text or filename"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
            </div>
          </div>
          <Select
            label="Model"
            className="w-56"
            value={modelFilter}
            onChange={(v) => {
              setModelFilter(v)
              setPage(0)
            }}
            options={[
              { value: '', label: 'All models' },
              ...(models.data?.models || []).map((m) => ({ value: m.id, label: m.name })),
            ]}
          />
          <label className="flex items-center gap-2 pb-2 text-sm">
            <input
              type="checkbox"
              className="h-4 w-4 rounded accent-indigo-600"
              checked={scoredOnly}
              onChange={(e) => {
                setScoredOnly(e.target.checked)
                setPage(0)
              }}
            />
            Scored only
          </label>
        </div>
      </Card>

      {list.error && <ErrorState error={list.error} onRetry={list.run} className="mb-6" />}

      <Card>
        <CardHeader title={`Entries (${count(total)})`} icon={Clock} />
        {list.loading ? (
          <Loading />
        ) : items.length === 0 ? (
          <EmptyState icon={FileText} title="No entries">
            {debounced || modelFilter || scoredOnly
              ? 'No entry matches these filters.'
              : 'Recognise an image and it will appear here.'}
          </EmptyState>
        ) : (
          <>
            <Table headers={['When', 'Text', 'Model', 'Confidence', 'Time', 'CER', '']}>
              {items.map((item) => (
                <tr
                  key={item.id}
                  className="cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/40"
                  onClick={() => setDetailId(item.id === detailId ? null : item.id)}
                >
                  <td className="px-3 py-2.5 text-xs text-slate-500" title={datetime(item.created_at)}>
                    {relativeTime(item.created_at)}
                  </td>
                  <td className="max-w-xs px-3 py-2.5">
                    <p className="truncate font-mono text-xs">{item.text || '—'}</p>
                    <p className="truncate text-[11px] text-slate-400">{item.original_filename}</p>
                  </td>
                  <td className="px-3 py-2.5 text-xs">{item.model_name}</td>
                  <td className={cx('nums px-3 py-2.5 text-xs', confidenceTone(item.confidence))}>
                    {percent(item.confidence)}
                  </td>
                  <td className="nums px-3 py-2.5 text-xs text-slate-500">{ms(item.total_ms)}</td>
                  <td className="px-3 py-2.5 text-xs">
                    {item.cer != null ? (
                      <span className={cx('nums', errorTone(item.cer))}>{rate(item.cer)}</span>
                    ) : (
                      <span className="text-slate-400">—</span>
                    )}
                  </td>
                  <td className="px-3 py-2.5 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        remove(item.id)
                      }}
                      className="rounded p-1 text-slate-400 hover:bg-rose-50 hover:text-rose-600 dark:hover:bg-rose-500/10"
                      aria-label="Delete entry"
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </td>
                </tr>
              ))}
            </Table>

            {pages > 1 && (
              <div className="flex items-center justify-between border-t border-slate-200 px-5 py-3 dark:border-slate-800">
                <span className="text-xs text-slate-500">
                  Page {page + 1} of {pages}
                </span>
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={page === 0}
                    onClick={() => setPage((p) => p - 1)}
                  >
                    Previous
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    disabled={page + 1 >= pages}
                    onClick={() => setPage((p) => p + 1)}
                  >
                    Next
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </Card>

      {detailId && <HistoryDetail id={detailId} onClose={() => setDetailId(null)} onChanged={refresh} />}
    </div>
  )
}

function HistoryDetail({ id, onClose, onChanged }) {
  const detail = useAsync(() => api.historyItem(id), [id])
  const [groundTruth, setGroundTruth] = useState('')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    if (detail.data) setGroundTruth(detail.data.ground_truth || '')
  }, [detail.data])

  const save = async () => {
    setSaving(true)
    try {
      const updated = await api.setGroundTruth(id, groundTruth)
      detail.setData(updated)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
      onChanged?.()
    } finally {
      setSaving(false)
    }
  }

  const item = detail.data

  return (
    <Card className="mt-6">
      <CardHeader
        title="Entry detail"
        subtitle={item ? `#${item.id} · ${datetime(item.created_at)}` : undefined}
        action={
          <Button size="sm" variant="ghost" icon={X} onClick={onClose}>
            Close
          </Button>
        }
      />
      {detail.loading && <Loading />}
      {detail.error && <ErrorState error={detail.error} />}
      {item && (
        <div className="grid gap-6 p-5 lg:grid-cols-2">
          <div className="space-y-4">
            {item.has_image ? (
              <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-100 dark:border-slate-800 dark:bg-slate-950">
                <img
                  src={api.historyImageUrl(item.id)}
                  alt={item.original_filename}
                  className="mx-auto max-h-80 w-auto object-contain"
                />
              </div>
            ) : (
              <EmptyState icon={ImageIcon} title="The image was not stored">
                This request was run with history saving disabled, or the image has since been
                pruned.
              </EmptyState>
            )}

            <dl className="grid grid-cols-2 gap-3 text-xs">
              <Field label="Model" value={item.model_name} />
              <Field label="Device" value={item.device} />
              <Field label="Decoding" value={item.decode_strategy} />
              <Field label="Image" value={`${item.image_width}×${item.image_height}`} />
              <Field label="Preprocessing" value={ms(item.preprocess_ms)} />
              <Field label="Segmentation" value={ms(item.segment_ms)} />
              <Field label="Inference" value={ms(item.inference_ms)} />
              <Field label="Total" value={ms(item.total_ms)} />
            </dl>
          </div>

          <div className="space-y-4">
            <div>
              <p className="label">Recognised text</p>
              <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-lg bg-slate-50 p-3 font-mono text-sm dark:bg-slate-950">
                {item.text || '(nothing)'}
              </pre>
            </div>

            <div>
              <label className="label" htmlFor="gt-edit">
                Ground truth
              </label>
              <textarea
                id="gt-edit"
                className="input min-h-24 font-mono"
                placeholder="Type what the image actually says, to score this result"
                value={groundTruth}
                onChange={(e) => setGroundTruth(e.target.value)}
              />
              <div className="mt-2 flex items-center gap-2">
                <Button size="sm" onClick={save} loading={saving} icon={saved ? Check : Target}>
                  {saved ? 'Saved' : 'Score this entry'}
                </Button>
                <span className="text-[11px] text-slate-500">
                  Leave empty and save to clear the score.
                </span>
              </div>
            </div>

            {item.cer != null && (
              <div className="grid grid-cols-3 gap-3">
                <Metric label="CER" value={rate(item.cer)} tone={errorTone(item.cer)} />
                <Metric label="WER" value={rate(item.wer)} tone={errorTone(item.wer)} />
                <Metric
                  label="Exact match"
                  value={item.exact_match ? 'Yes' : 'No'}
                  tone={item.exact_match ? 'text-emerald-600' : 'text-slate-500'}
                />
              </div>
            )}

            {item.lines?.length > 1 && (
              <div>
                <p className="label">Per line</p>
                <ul className="max-h-48 space-y-1 overflow-auto">
                  {item.lines.map((line) => (
                    <li
                      key={line.index}
                      className="flex items-center gap-2 rounded bg-slate-50 px-2 py-1 text-xs dark:bg-slate-950"
                    >
                      <span className="nums w-5 text-right text-slate-400">{line.index + 1}</span>
                      <span className="min-w-0 flex-1 truncate font-mono">{line.text}</span>
                      <Badge tone={line.confidence >= 0.8 ? 'emerald' : 'amber'}>
                        {percent(line.confidence, 0)}
                      </Badge>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </Card>
  )
}

function Field({ label, value }) {
  return (
    <div>
      <dt className="text-[11px] text-slate-500">{label}</dt>
      <dd className="nums mt-0.5 font-medium">{value}</dd>
    </div>
  )
}

function Metric({ label, value, tone }) {
  return (
    <div className="rounded-lg border border-slate-200 px-3 py-2 text-center dark:border-slate-800">
      <p className="text-[11px] text-slate-500">{label}</p>
      <p className={cx('nums mt-0.5 text-sm font-semibold', tone)}>{value}</p>
    </div>
  )
}
