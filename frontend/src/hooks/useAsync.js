import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Run an async function and track {data, error, loading}.
 *
 * Guards against the two bugs that this pattern always produces: setting
 * state after unmount, and an earlier slow request overwriting a later fast
 * one. Each run gets a sequence number and only the newest may write.
 *
 * @param {Function} fn        the async function
 * @param {Array} deps         re-run when these change
 * @param {object} options
 * @param {boolean} options.immediate  run on mount (default true)
 */
export function useAsync(fn, deps = [], { immediate = true } = {}) {
  const [state, setState] = useState({ data: null, error: null, loading: immediate })
  const mounted = useRef(true)
  const sequence = useRef(0)

  useEffect(() => {
    mounted.current = true
    return () => {
      mounted.current = false
    }
  }, [])

  const run = useCallback(
    async (...args) => {
      const id = ++sequence.current
      setState((previous) => ({ ...previous, loading: true, error: null }))
      try {
        const data = await fn(...args)
        if (mounted.current && id === sequence.current) {
          setState({ data, error: null, loading: false })
        }
        return data
      } catch (error) {
        if (error.name === 'AbortError') return undefined
        if (mounted.current && id === sequence.current) {
          setState({ data: null, error, loading: false })
        }
        return undefined
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    deps,
  )

  useEffect(() => {
    if (immediate) run()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return { ...state, run, setData: (data) => setState((s) => ({ ...s, data })) }
}

/**
 * Poll `fn` every `intervalMs` while `active` is true.
 *
 * Used for training-job progress. Polling stops as soon as `active` goes
 * false, so a finished job does not keep the server busy.
 */
export function usePolling(fn, { intervalMs = 2000, active = true } = {}) {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const saved = useRef(fn)

  useEffect(() => {
    saved.current = fn
  }, [fn])

  useEffect(() => {
    if (!active) return undefined

    let cancelled = false

    const tick = async () => {
      try {
        const result = await saved.current()
        if (!cancelled) {
          setData(result)
          setError(null)
        }
      } catch (caught) {
        if (!cancelled) setError(caught)
      }
    }

    tick()
    const handle = setInterval(tick, intervalMs)
    return () => {
      cancelled = true
      clearInterval(handle)
    }
  }, [active, intervalMs])

  return { data, error }
}
