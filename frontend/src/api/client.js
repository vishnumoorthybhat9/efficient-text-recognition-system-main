/**
 * API client.
 *
 * Every call funnels through `request`, which turns the backend's structured
 * error envelope ({error, message, detail, request_id}) into an `ApiError`
 * carrying the human-readable `message`. The UI shows that message verbatim,
 * which is why the backend writes it for a person to read.
 */

const BASE = import.meta.env.VITE_API_BASE || '/api'

export class ApiError extends Error {
  constructor(message, { status, code, detail, requestId } = {}) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.detail = detail || {}
    this.requestId = requestId
  }
}

async function request(path, { method = 'GET', body, headers, signal } = {}) {
  let response
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      body,
      headers,
      signal,
    })
  } catch (cause) {
    if (cause.name === 'AbortError') throw cause
    // fetch only rejects for network-level failures, which almost always mean
    // the Python server is not running.
    throw new ApiError(
      'Cannot reach the backend. Start it with:  python scripts/run_api.py',
      { code: 'network_error' },
    )
  }

  if (response.status === 204) return null

  const contentType = response.headers.get('content-type') || ''
  const isJson = contentType.includes('application/json')
  const payload = isJson ? await response.json().catch(() => null) : await response.text()

  if (!response.ok) {
    const message =
      (payload && payload.message) ||
      (typeof payload === 'string' && payload.slice(0, 300)) ||
      `Request failed with status ${response.status}`
    throw new ApiError(message, {
      status: response.status,
      code: payload?.error,
      detail: payload?.detail,
      requestId: payload?.request_id,
    })
  }

  return payload
}

function query(params = {}) {
  const search = new URLSearchParams()
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== '') {
      search.append(key, String(value))
    }
  })
  const text = search.toString()
  return text ? `?${text}` : ''
}

export const api = {
  // --- system ---
  health: () => request('/health'),
  systemInfo: () => request('/system/info'),

  // --- models ---
  models: () => request('/models'),
  model: (id) => request(`/models/${encodeURIComponent(id)}`),
  loadModel: (id) => request(`/models/${encodeURIComponent(id)}/load`, { method: 'POST' }),
  unloadModel: (id) => request(`/models/${encodeURIComponent(id)}/unload`, { method: 'POST' }),
  refreshModels: () => request('/models/refresh', { method: 'POST' }),

  // --- recognition ---
  recognize: ({ file, modelId, options, groundTruth, signal }) => {
    const form = new FormData()
    form.append('file', file)
    if (modelId) form.append('model_id', modelId)
    if (options) form.append('options', JSON.stringify(options))
    if (groundTruth) form.append('ground_truth', groundTruth)
    return request('/ocr/recognize', { method: 'POST', body: form, signal })
  },

  compare: ({ file, modelIds, options, groundTruth, signal }) => {
    const form = new FormData()
    form.append('file', file)
    form.append('model_ids', modelIds.join(','))
    if (options) form.append('options', JSON.stringify(options))
    if (groundTruth) form.append('ground_truth', groundTruth)
    return request('/ocr/compare', { method: 'POST', body: form, signal })
  },

  preprocessPreview: ({ file, options, detectLines = true, signal }) => {
    const form = new FormData()
    form.append('file', file)
    if (options) form.append('options', JSON.stringify(options))
    form.append('detect_lines', String(detectLines))
    return request('/ocr/preprocess/preview', { method: 'POST', body: form, signal })
  },

  // --- export ---
  // The server writes exactly the text it is given, so an edited
  // transcription downloads as edited.
  exportFile: async (format, payload) => {
    const response = await fetch(`${BASE}/ocr/export/${format}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!response.ok) {
      const detail = await response.json().catch(() => null)
      throw new ApiError(detail?.message || `Export failed (${response.status})`, {
        status: response.status,
        code: detail?.error,
      })
    }
    return response.blob()
  },

  // --- history ---
  history: (params) => request(`/history${query(params)}`),
  historyItem: (id) => request(`/history/${id}`),
  historyStats: () => request('/history/stats'),
  historyImageUrl: (id) => `${BASE}/history/${id}/image`,
  setGroundTruth: (id, groundTruth) =>
    request(`/history/${id}/ground-truth`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ground_truth: groundTruth }),
    }),
  deleteHistoryItem: (id) => request(`/history/${id}`, { method: 'DELETE' }),
  clearHistory: () => request(`/history${query({ confirm: true })}`, { method: 'DELETE' }),

  // --- evaluation ---
  evaluationRuns: (params) => request(`/evaluation/runs${query(params)}`),
  evaluationRun: (id) => request(`/evaluation/runs/${id}`),
  comparison: (params) => request(`/evaluation/comparison${query(params)}`),
  runEvaluation: (payload) =>
    request('/evaluation/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  trainingHistory: (runName) => request(`/evaluation/training-history/${encodeURIComponent(runName)}`),
  charts: () => request('/evaluation/charts'),
  chart: (id, params) => request(`/evaluation/charts/${id}${query(params)}`),

  // --- datasets ---
  datasets: () => request('/datasets'),
  dataset: (name, params) => request(`/datasets/${encodeURIComponent(name)}${query(params)}`),
  demoImages: () => request('/datasets/demo'),
  demoImageUrl: (file) => `${BASE}/datasets/demo/${encodeURIComponent(file)}`,

  // --- training ---
  availableJobs: () => request('/training/jobs/available'),
  trainingJobs: () => request('/training/jobs'),
  trainingJob: (id) => request(`/training/jobs/${id}`),
  trainingJobLog: (id, lines = 200) => request(`/training/jobs/${id}/log${query({ lines })}`),
  startTraining: (payload) =>
    request('/training/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  cancelTraining: (id) => request(`/training/jobs/${id}/cancel`, { method: 'POST' }),
}

/** Fetch a demo image and wrap it in a File, so it can be uploaded like any other. */
export async function demoImageAsFile(filename) {
  const response = await fetch(api.demoImageUrl(filename))
  if (!response.ok) throw new ApiError(`Could not load the demo image ${filename}`)
  const blob = await response.blob()
  return new File([blob], filename, { type: blob.type || 'image/png' })
}
