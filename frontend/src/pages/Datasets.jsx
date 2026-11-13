import { useState } from 'react'
import { Check, Database, Download, ExternalLink, FolderOpen, Images, Info } from 'lucide-react'
import { api } from '../api/client.js'
import { useAsync } from '../hooks/useAsync.js'
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
  Select,
  Table,
} from '../components/ui.jsx'
import { count, datetime } from '../lib/format.js'

export default function Datasets() {
  const datasets = useAsync(() => api.datasets(), [])
  const demos = useAsync(() => api.demoImages(), [])
  const [inspecting, setInspecting] = useState(null)

  const prepared = datasets.data?.prepared || []
  const external = datasets.data?.external || []

  return (
    <div>
      <PageHeader
        title="Datasets"
        description="What data is on this machine, and how to obtain the public benchmarks. None of the external datasets are redistributed with this project."
      />

      {datasets.error && <ErrorState error={datasets.error} onRetry={datasets.run} className="mb-6" />}

      {datasets.data && prepared.length === 0 && (
        <Alert tone="warning" title="No dataset is prepared" className="mb-6">
          Generate a synthetic corpus from your system fonts — no download needed:
          <code className="mt-2 block font-mono text-xs">python scripts/make_synthetic_data.py</code>
        </Alert>
      )}

      {/* --- prepared --- */}
      <Card className="mb-6">
        <CardHeader
          title="Prepared datasets"
          subtitle={datasets.data ? `In ${datasets.data.data_root}` : undefined}
          icon={FolderOpen}
        />
        {datasets.loading ? (
          <Loading />
        ) : prepared.length === 0 ? (
          <EmptyState icon={Database} title="Nothing prepared yet" />
        ) : (
          <div className="divide-y divide-slate-100 dark:divide-slate-800">
            {prepared.map((dataset) => (
              <div key={dataset.name} className="px-5 py-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <h3 className="text-sm font-semibold">{dataset.name}</h3>
                      {dataset.benchmark_only ? (
                        <Badge tone="blue">Evaluation only</Badge>
                      ) : dataset.trainable ? (
                        <Badge tone="emerald">
                          <Check className="mr-1 h-3 w-3" />
                          Trainable
                        </Badge>
                      ) : (
                        <Badge tone="amber">Incomplete</Badge>
                      )}
                      <Badge tone="slate">{dataset.source}</Badge>
                    </div>
                    <p className="mt-1 text-[11px] text-slate-500">
                      {dataset.path}
                      {dataset.generated_at && ` · prepared ${datetime(dataset.generated_at)}`}
                      {dataset.seed != null && ` · seed ${dataset.seed}`}
                    </p>
                  </div>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={() => setInspecting(inspecting === dataset.name ? null : dataset.name)}
                  >
                    {inspecting === dataset.name ? 'Hide samples' : 'Inspect'}
                  </Button>
                </div>

                <div className="mt-3 flex flex-wrap gap-2">
                  {dataset.splits.map((split) => (
                    <span
                      key={split.split}
                      className={`rounded-lg px-2.5 py-1 text-[11px] ${
                        split.available
                          ? 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300'
                          : 'bg-slate-50 text-slate-400 dark:bg-slate-900'
                      }`}
                    >
                      {split.split}: <span className="nums font-medium">{count(split.count)}</span>
                    </span>
                  ))}
                  {dataset.unlabeled?.available && (
                    <span className="rounded-lg bg-indigo-50 px-2.5 py-1 text-[11px] text-indigo-700 dark:bg-indigo-500/10 dark:text-indigo-300">
                      unlabeled (SimCLR):{' '}
                      <span className="nums font-medium">{count(dataset.unlabeled.count)}</span>
                    </span>
                  )}
                </div>

                {dataset.notes && (
                  <p className="mt-2 text-[11px] leading-relaxed text-slate-500">{dataset.notes}</p>
                )}

                {inspecting === dataset.name && (
                  <DatasetSamples
                    key={dataset.name}
                    name={dataset.name}
                    benchmarkOnly={dataset.benchmark_only}
                  />
                )}
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* --- demo images --- */}
      {demos.data?.images?.length > 0 && (
        <Card className="mb-6">
          <CardHeader
            title="Demo images"
            subtitle="Bundled samples so the app is usable with no upload and no dataset"
            icon={Images}
          />
          <div className="grid gap-3 p-5 sm:grid-cols-2 lg:grid-cols-4">
            {demos.data.images.map((demo) => (
              <div
                key={demo.file}
                className="overflow-hidden rounded-lg border border-slate-200 dark:border-slate-800"
              >
                <img
                  src={api.demoImageUrl(demo.file)}
                  alt={demo.text}
                  className="h-20 w-full bg-slate-100 object-contain dark:bg-slate-950"
                />
                <div className="px-3 py-2">
                  <p className="truncate font-mono text-[11px]">{demo.text}</p>
                  <Badge tone={demo.kind === 'handwritten' ? 'amber' : 'blue'}>{demo.kind}</Badge>
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* --- external --- */}
      <Card>
        <CardHeader
          title="Public benchmark datasets"
          subtitle="The datasets named in the project synopsis, with official sources"
          icon={Download}
        />
        <div className="divide-y divide-slate-100 dark:divide-slate-800">
          {external.map((dataset) => (
            <div key={dataset.key} className="px-5 py-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="text-sm font-semibold">{dataset.name}</h3>
                    <Badge tone="slate">{dataset.size}</Badge>
                    {dataset.registration_required && <Badge tone="amber">Registration required</Badge>}
                  </div>
                  <p className="mt-1 text-[13px] text-slate-600 dark:text-slate-400">
                    {dataset.content}
                  </p>
                  <p className="mt-1 text-[11px] text-slate-500">{dataset.licence}</p>
                </div>
                <a href={dataset.url} target="_blank" rel="noreferrer noopener">
                  <Button size="sm" variant="secondary" icon={ExternalLink}>
                    Official page
                  </Button>
                </a>
              </div>
              <pre className="mt-3 overflow-x-auto rounded-lg bg-slate-50 px-3 py-2 font-mono text-[11px] leading-relaxed text-slate-600 dark:bg-slate-950 dark:text-slate-400">
                {dataset.preparation}
              </pre>
            </div>
          ))}
        </div>
        <div className="border-t border-slate-200 px-5 py-3 dark:border-slate-800">
          <p className="flex items-start gap-2 text-[11px] text-slate-500">
            <Info className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            Once a dataset is imported it uses the same manifest format as everything else, so
            training and evaluation run against it unchanged.
          </p>
        </div>
      </Card>
    </div>
  )
}

function DatasetSamples({ name, benchmarkOnly = false }) {
  const [split, setSplit] = useState(benchmarkOnly ? 'test' : 'train')
  const detail = useAsync(() => api.dataset(name, { split, limit: 12 }), [name, split])

  return (
    <div className="mt-4 rounded-lg border border-slate-200 p-4 dark:border-slate-800">
      <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
        <Select
          label="Split"
          className="w-36"
          value={split}
          onChange={setSplit}
          options={
            benchmarkOnly
              ? [{ value: 'test', label: 'test' }]
              : [
                  { value: 'train', label: 'train' },
                  { value: 'val', label: 'val' },
                  { value: 'test', label: 'test' },
                ]
          }
        />
        {detail.data && (
          <p className="nums pb-2 text-[11px] text-slate-500">
            {count(detail.data.count)} samples · label length{' '}
            {detail.data.label_length.min}–{detail.data.label_length.max} (mean{' '}
            {detail.data.label_length.mean})
          </p>
        )}
      </div>

      {detail.loading && <Loading label="Loading samples…" />}
      {detail.error && <ErrorState error={detail.error} />}

      {detail.data && (
        <>
          {Object.keys(detail.data.unsupported_characters || {}).length > 0 && (
            <Alert tone="warning" title="Characters outside the charset" className="mb-3">
              These appear in the labels but cannot be represented, so those samples are dropped
              during training:{' '}
              <code className="font-mono">
                {Object.keys(detail.data.unsupported_characters).join(' ')}
              </code>
            </Alert>
          )}
          <Table headers={['Sample id', 'Label']}>
            {detail.data.samples.map((sample) => (
              <tr key={sample.id}>
                <td className="px-3 py-1.5 font-mono text-[11px] text-slate-500">{sample.id}</td>
                <td className="px-3 py-1.5 font-mono text-xs">{sample.label}</td>
              </tr>
            ))}
          </Table>
        </>
      )}
    </div>
  )
}
