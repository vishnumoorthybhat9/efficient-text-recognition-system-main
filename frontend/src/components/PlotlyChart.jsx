import { useEffect, useRef, useState } from 'react'
import { BarChart3 } from 'lucide-react'
import { EmptyState, Loading } from './ui.jsx'

/**
 * Render a Plotly figure built by the backend.
 *
 * Plotly is imported dynamically: the bundle is ~3 MB, and pages without a
 * chart should not pay for it. The library is also driven directly rather
 * than through react-plotly.js — that wrapper lags React releases, and this
 * is about thirty lines.
 */
export function PlotlyChart({ figure, height = 380, className }) {
  const container = useRef(null)
  const [ready, setReady] = useState(false)
  const [failed, setFailed] = useState(null)

  useEffect(() => {
    let cancelled = false
    let plotly = null

    async function draw() {
      if (!container.current || !figure) return
      try {
        const module = await import('plotly.js-dist-min')
        plotly = module.default || module
        if (cancelled || !container.current) return

        const isDark = document.documentElement.classList.contains('dark')
        const layout = {
          ...figure.layout,
          height,
          autosize: true,
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          font: {
            ...(figure.layout?.font || {}),
            color: isDark ? '#cbd5e1' : '#334155',
          },
        }
        // Grid lines from the plotly_white template are invisible on a dark
        // card, so the axis colours are overridden per theme.
        for (const axis of ['xaxis', 'yaxis', 'yaxis2']) {
          if (layout[axis]) {
            layout[axis] = {
              ...layout[axis],
              gridcolor: isDark ? 'rgba(148,163,184,0.15)' : 'rgba(100,116,139,0.15)',
              linecolor: isDark ? 'rgba(148,163,184,0.3)' : 'rgba(100,116,139,0.3)',
              zerolinecolor: isDark ? 'rgba(148,163,184,0.2)' : 'rgba(100,116,139,0.2)',
            }
          }
        }

        await plotly.react(container.current, figure.data || [], layout, {
          responsive: true,
          displaylogo: false,
          modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
          toImageButtonOptions: { format: 'png', scale: 2 },
        })
        if (!cancelled) setReady(true)
      } catch (error) {
        if (!cancelled) setFailed(error)
      }
    }

    draw()
    const node = container.current

    return () => {
      cancelled = true
      if (plotly && node) plotly.purge(node)
    }
  }, [figure, height])

  if (failed) {
    return (
      <EmptyState icon={BarChart3} title="The chart could not be rendered">
        {failed.message}
      </EmptyState>
    )
  }

  return (
    <div className={className} style={{ minHeight: height }}>
      {!ready && <Loading label="Drawing chart…" />}
      <div ref={container} style={{ width: '100%', display: ready ? 'block' : 'none' }} />
    </div>
  )
}

/** A chart payload from the API, including its `empty` state. */
export function ChartPanel({ chart, height, className }) {
  if (!chart) return <Loading label="Loading chart…" />

  if (chart.empty) {
    return (
      <EmptyState icon={BarChart3} title={chart.title} className={className}>
        {chart.empty_reason}
      </EmptyState>
    )
  }

  return (
    <div className={className}>
      <PlotlyChart figure={chart.figure} height={height} />
      {chart.description && (
        <p className="px-5 pb-4 text-xs leading-relaxed text-slate-500 dark:text-slate-400">
          {chart.description}
        </p>
      )}
    </div>
  )
}
