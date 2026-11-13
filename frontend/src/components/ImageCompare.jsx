import { useEffect, useRef, useState } from 'react'
import { Images, SplitSquareHorizontal } from 'lucide-react'
import { api } from '../api/client.js'
import { Alert, Button, Card, CardHeader, Loading } from './ui.jsx'
import { cx, ms } from '../lib/format.js'

/**
 * Original image beside what the model actually saw.
 *
 * Preprocessing is not always an improvement -- denoising a clean scan eats
 * thin strokes, and binarising an unevenly lit photograph can erase half the
 * text. Showing both side by side is the only honest way to present it, and it
 * is usually enough to explain a bad result at a glance.
 *
 * The original file is never modified; the processed view is rendered by the
 * server from a copy.
 */
export function ImageCompare({ file, options, className }) {
  const [processed, setProcessed] = useState(null)
  const [meta, setMeta] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState('side')
  const [slider, setSlider] = useState(50)
  const [originalUrl, setOriginalUrl] = useState(null)
  const abort = useRef(null)

  useEffect(() => {
    if (!file) {
      setOriginalUrl(null)
      return undefined
    }
    const url = URL.createObjectURL(file)
    setOriginalUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  useEffect(() => {
    if (!file) {
      setProcessed(null)
      return undefined
    }

    abort.current?.abort()
    abort.current = new AbortController()
    setLoading(true)
    setError(null)

    api
      .preprocessPreview({
        file,
        options: { preprocess: options?.preprocess },
        detectLines: false,
        signal: abort.current.signal,
      })
      .then((data) => {
        // The last captured stage is the image the recogniser receives.
        const stages = data.stages || []
        setProcessed(stages.length ? stages[stages.length - 1] : null)
        setMeta(data.metadata || null)
      })
      .catch((caught) => {
        if (caught.name !== 'AbortError') setError(caught)
      })
      .finally(() => setLoading(false))

    return () => abort.current?.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [file, JSON.stringify(options?.preprocess)])

  if (!file) return null

  return (
    <Card className={className}>
      <CardHeader
        title="Original and processed"
        subtitle={
          meta
            ? `${meta.steps?.join(' → ') || 'no changes'} · ${ms(meta.total_ms)}`
            : 'What the model actually sees'
        }
        icon={Images}
        action={
          <Button
            size="sm"
            variant="secondary"
            icon={SplitSquareHorizontal}
            onClick={() => setMode(mode === 'side' ? 'slider' : 'side')}
          >
            {mode === 'side' ? 'Overlay' : 'Side by side'}
          </Button>
        }
      />

      {loading && <Loading label="Rendering the processed image…" />}
      {error && (
        <div className="p-5">
          <Alert tone="warning" title="Could not render the processed image">
            {error.message}
          </Alert>
        </div>
      )}

      {!loading && !error && processed && (
        <div className="p-5">
          {mode === 'side' ? (
            <div className="grid gap-4 sm:grid-cols-2">
              {[
                ['Original (unmodified)', originalUrl],
                ['Processed — what the model sees', processed.image],
              ].map(([label, src]) => (
                <figure key={label}>
                  <figcaption className="mb-1.5 text-[11px] font-medium text-slate-500">
                    {label}
                  </figcaption>
                  <div className="overflow-hidden rounded-lg border border-slate-200 bg-slate-100 dark:border-slate-800 dark:bg-slate-950">
                    <img src={src} alt={label} className="mx-auto max-h-72 w-auto object-contain" />
                  </div>
                </figure>
              ))}
            </div>
          ) : (
            <div>
              <div className="relative overflow-hidden rounded-lg border border-slate-200 bg-slate-100 dark:border-slate-800 dark:bg-slate-950">
                <img src={originalUrl} alt="Original" className="mx-auto max-h-80 w-auto object-contain" />
                <div
                  className="absolute inset-0 overflow-hidden"
                  style={{ clipPath: `inset(0 0 0 ${slider}%)` }}
                >
                  <img
                    src={processed.image}
                    alt="Processed"
                    className="mx-auto h-full w-auto object-contain"
                  />
                </div>
                <div
                  className="pointer-events-none absolute inset-y-0 w-0.5 bg-indigo-500"
                  style={{ left: `${slider}%` }}
                />
              </div>
              <input
                type="range"
                min={0}
                max={100}
                value={slider}
                onChange={(e) => setSlider(Number(e.target.value))}
                className="mt-3 h-1.5 w-full cursor-pointer appearance-none rounded-full bg-slate-200 accent-indigo-600 dark:bg-slate-800"
                aria-label="Compare original and processed"
              />
              <div className="mt-1 flex justify-between text-[11px] text-slate-500">
                <span>Original</span>
                <span>Processed</span>
              </div>
            </div>
          )}

          {meta && (
            <dl className={cx('mt-4 grid grid-cols-2 gap-2 text-[11px] sm:grid-cols-4')}>
              {[
                ['Output size', `${meta.width}×${meta.height}`],
                ['Deskewed', meta.skew_angle_deg ? `${meta.skew_angle_deg.toFixed(2)}°` : 'not needed'],
                ['Polarity', meta.inverted ? 'inverted' : 'unchanged'],
                ['Perspective', meta.perspective_corrected ? 'corrected' : 'not applied'],
              ].map(([label, value]) => (
                <div key={label} className="rounded-lg bg-slate-50 px-2.5 py-1.5 dark:bg-slate-950">
                  <dt className="text-slate-500">{label}</dt>
                  <dd className="nums mt-0.5 font-medium">{value}</dd>
                </div>
              ))}
            </dl>
          )}

          <p className="mt-3 text-[11px] text-slate-500">
            Your original file is never modified. If the processed image looks worse than the
            original, turn steps off — preprocessing can hurt as easily as help.
          </p>
        </div>
      )}
    </Card>
  )
}
