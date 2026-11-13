import { useRef, useState } from 'react'
import { Eye, Images, Play, ScanLine, Sliders } from 'lucide-react'
import { api, demoImageAsFile } from '../api/client.js'
import { useAsync } from '../hooks/useAsync.js'
import { PageHeader } from '../components/Layout.jsx'
import { FileDrop } from '../components/FileDrop.jsx'
import { DEFAULT_PREPROCESS, PreprocessControls } from '../components/PreprocessControls.jsx'
import {
  Alert,
  Badge,
  Button,
  Card,
  CardHeader,
  EmptyState,
  ErrorState,
  Loading,
  Toggle,
} from '../components/ui.jsx'
import { ms } from '../lib/format.js'

/**
 * Shows what each enhancement step actually did to the image.
 *
 * The point is accountability: toggles that claim to "enhance" an image are
 * worth nothing if you cannot see the result, and several of these steps make
 * recognition *worse* on clean input.
 */
export default function PreprocessLab() {
  const [file, setFile] = useState(null)
  const [options, setOptions] = useState(DEFAULT_PREPROCESS)
  const [detectLines, setDetectLines] = useState(true)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const abort = useRef(null)

  const demos = useAsync(() => api.demoImages(), [])

  const run = async () => {
    if (!file) return
    abort.current?.abort()
    abort.current = new AbortController()
    setBusy(true)
    setError(null)
    try {
      const data = await api.preprocessPreview({
        file,
        options: { preprocess: options },
        detectLines,
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
      setResult(null)
    } catch (caught) {
      setError(caught)
    }
  }

  return (
    <div>
      <PageHeader
        title="Preprocessing Lab"
        description="Apply the enhancement pipeline and inspect every intermediate image. These are the same OpenCV operations the recognition pipeline runs."
      />

      <div className="grid gap-6 lg:grid-cols-[340px_minmax(0,1fr)]">
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
            </div>
          </Card>

          <Card>
            <CardHeader title="Pipeline steps" icon={Sliders} />
            <div className="space-y-5 p-5">
              <PreprocessControls value={options} onChange={setOptions} disabled={busy} compact />
              <div className="border-t border-slate-200 pt-4 dark:border-slate-800">
                <Toggle
                  label="Detect text lines"
                  description="Run line segmentation on the enhanced image"
                  checked={detectLines}
                  onChange={setDetectLines}
                  disabled={busy}
                />
              </div>
              <Button className="w-full" icon={Play} onClick={run} loading={busy} disabled={!file}>
                Run pipeline
              </Button>
              <Button
                className="w-full"
                variant="ghost"
                size="sm"
                onClick={() => setOptions(DEFAULT_PREPROCESS)}
                disabled={busy}
              >
                Reset to defaults
              </Button>
            </div>
          </Card>
        </div>

        <div className="space-y-6">
          {error && <ErrorState error={error} onRetry={run} />}

          {!result && !busy && (
            <Card>
              <EmptyState icon={Eye} title="Nothing to show yet">
                Choose an image, pick some steps and run the pipeline. Each stage will appear here
                as an image, with the time it took.
              </EmptyState>
            </Card>
          )}

          {busy && (
            <Card>
              <Loading label="Running the pipeline…" />
            </Card>
          )}

          {result && (
            <>
              <Card>
                <CardHeader
                  title={`Stages (${result.stages.length})`}
                  subtitle={`Total ${ms(result.metadata.total_ms)} · output ${result.metadata.width}×${result.metadata.height} px${
                    result.metadata.inverted ? ' · polarity was inverted' : ''
                  }${
                    result.metadata.skew_angle_deg
                      ? ` · deskewed ${result.metadata.skew_angle_deg.toFixed(2)}°`
                      : ''
                  }`}
                />
                <div className="space-y-4 p-5">
                  {result.stages.map((stage, index) => (
                    <div key={`${stage.name}-${index}`}>
                      <div className="mb-1.5 flex flex-wrap items-center gap-2">
                        <span className="nums flex h-5 w-5 items-center justify-center rounded bg-slate-200 text-[10px] font-semibold text-slate-600 dark:bg-slate-800 dark:text-slate-300">
                          {index}
                        </span>
                        <span className="text-sm font-medium">{stage.label}</span>
                        {stage.detail && <Badge tone="slate">{stage.detail}</Badge>}
                        {stage.duration_ms > 0 && (
                          <span className="nums text-[11px] text-slate-500">
                            {ms(stage.duration_ms)}
                          </span>
                        )}
                      </div>
                      <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-100 dark:border-slate-800 dark:bg-slate-950">
                        <img
                          src={stage.image}
                          alt={stage.label}
                          className="mx-auto max-h-60 w-auto object-contain"
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </Card>

              {result.overlay && (
                <Card>
                  <CardHeader
                    title={`Detected text lines (${result.detected_lines.length})`}
                    subtitle="Each box is recognised independently and the results are joined with newlines."
                    icon={ScanLine}
                  />
                  <div className="p-5">
                    <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-100 dark:border-slate-800 dark:bg-slate-950">
                      <img src={result.overlay} alt="Detected lines" className="mx-auto w-full object-contain" />
                    </div>
                    {result.detected_lines.length === 0 && (
                      <Alert tone="warning" className="mt-3">
                        No line was detected. The whole image will be treated as a single line.
                      </Alert>
                    )}
                  </div>
                </Card>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
