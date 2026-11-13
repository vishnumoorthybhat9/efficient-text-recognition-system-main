import { Sliders } from 'lucide-react'
import { Select, Slider, Toggle } from './ui.jsx'

/** The defaults the backend uses when no options are sent. */
export const DEFAULT_PREPROCESS = {
  grayscale: true,
  deskew: true,
  perspective: false,
  denoise: false,
  denoise_strength: 7,
  denoise_method: 'nlmeans',
  clahe: false,
  clahe_clip_limit: 2.0,
  sharpen: false,
  sharpen_amount: 1.0,
  binarize: 'none',
  binarize_block_size: 31,
  binarize_offset: 10,
  invert_if_dark: true,
  normalize: true,
}

export const DEFAULT_RECOGNIZE_OPTIONS = {
  single_line: false,
  decode_strategy: 'greedy',
  beam_width: 10,
  max_lines: 200,
  min_confidence: 0,
  handwritten: true,
  line_detector: 'auto',
  preprocess: DEFAULT_PREPROCESS,
  save_to_history: true,
}

/**
 * Controls for the image-enhancement pipeline.
 *
 * Every toggle maps one-to-one onto a real OpenCV step in
 * `ml/preprocessing/ops.py`; the Preprocessing Lab shows what each one did.
 * They all default to off because each can hurt as well as help -- denoising
 * a clean scan removes thin strokes, and binarising a photograph with uneven
 * lighting can erase half the text.
 */
export function PreprocessControls({ value, onChange, disabled = false, compact = false }) {
  const set = (patch) => onChange({ ...value, ...patch })

  return (
    <div className="space-y-4">
      {!compact && (
        <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
          <Sliders className="h-3.5 w-3.5" />
          Image enhancement
        </div>
      )}

      <div className="space-y-3">
        <Toggle
          label="Normalise polarity"
          description="Flip light-on-dark text so it is dark on light"
          checked={value.invert_if_dark}
          onChange={(v) => set({ invert_if_dark: v })}
          disabled={disabled}
        />
        <Toggle
          label="Deskew"
          description="Straighten rotated text"
          checked={value.deskew}
          onChange={(v) => set({ deskew: v })}
          disabled={disabled}
        />
        <Toggle
          label="Perspective correction"
          description="Flatten a photographed page shot at an angle"
          checked={value.perspective}
          onChange={(v) => set({ perspective: v })}
          disabled={disabled}
        />
        <Toggle
          label="Denoise"
          description="Remove scan and sensor noise"
          checked={value.denoise}
          onChange={(v) => set({ denoise: v })}
          disabled={disabled}
        />
        {value.denoise && (
          <div className="ml-12 space-y-3 border-l border-slate-200 pl-4 dark:border-slate-800">
            <Select
              label="Method"
              value={value.denoise_method}
              onChange={(v) => set({ denoise_method: v })}
              disabled={disabled}
              options={[
                { value: 'nlmeans', label: 'Non-local means (best quality)' },
                { value: 'bilateral', label: 'Bilateral (fast, keeps edges)' },
                { value: 'median', label: 'Median (salt-and-pepper)' },
              ]}
            />
            <Slider
              label="Strength"
              min={1}
              max={20}
              value={value.denoise_strength}
              onChange={(v) => set({ denoise_strength: v })}
            />
          </div>
        )}

        <Toggle
          label="Enhance contrast (CLAHE)"
          description="Recover text from unevenly lit photographs"
          checked={value.clahe}
          onChange={(v) => set({ clahe: v })}
          disabled={disabled}
        />
        {value.clahe && (
          <div className="ml-12 border-l border-slate-200 pl-4 dark:border-slate-800">
            <Slider
              label="Clip limit"
              min={0.5}
              max={6}
              step={0.5}
              value={value.clahe_clip_limit}
              onChange={(v) => set({ clahe_clip_limit: v })}
              format={(v) => v.toFixed(1)}
            />
          </div>
        )}

        <Toggle
          label="Sharpen"
          description="Unsharp mask, for softly focused scans"
          checked={value.sharpen}
          onChange={(v) => set({ sharpen: v })}
          disabled={disabled}
        />
        {value.sharpen && (
          <div className="ml-12 border-l border-slate-200 pl-4 dark:border-slate-800">
            <Slider
              label="Amount"
              min={0.2}
              max={2.5}
              step={0.1}
              value={value.sharpen_amount}
              onChange={(v) => set({ sharpen_amount: v })}
              format={(v) => v.toFixed(1)}
            />
          </div>
        )}
      </div>

      <div className="border-t border-slate-200 pt-4 dark:border-slate-800">
        <Select
          label="Binarisation"
          value={value.binarize}
          onChange={(v) => set({ binarize: v })}
          disabled={disabled}
          options={[
            { value: 'none', label: 'None (keep grayscale)' },
            { value: 'otsu', label: 'Otsu — global, for clean scans' },
            { value: 'adaptive', label: 'Adaptive — local mean' },
            { value: 'sauvola', label: 'Sauvola — best for degraded documents' },
          ]}
        />
        {value.binarize !== 'none' && value.binarize !== 'otsu' && (
          <div className="mt-3 space-y-3">
            <Slider
              label="Window size"
              min={5}
              max={101}
              step={2}
              value={value.binarize_block_size}
              onChange={(v) => set({ binarize_block_size: v })}
              format={(v) => `${v} px`}
            />
            {value.binarize === 'adaptive' && (
              <Slider
                label="Threshold offset"
                min={-20}
                max={40}
                value={value.binarize_offset}
                onChange={(v) => set({ binarize_offset: v })}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

/** Decoding and segmentation controls. */
export function RecognitionControls({ value, onChange, supportsBeam = true, disabled = false }) {
  const set = (patch) => onChange({ ...value, ...patch })

  return (
    <div className="space-y-4">
      <Toggle
        label="Single line"
        description="Skip line detection — the image is already one cropped line"
        checked={value.single_line}
        onChange={(v) => set({ single_line: v })}
        disabled={disabled}
      />

      <Toggle
        label="Handwritten document"
        description="Notebook pages, letters, prescriptions — removes shadows and ruled lines, then finds handwritten lines"
        checked={value.handwritten}
        onChange={(v) => set({ handwritten: v })}
        disabled={disabled || value.single_line}
      />

      {value.handwritten && !value.single_line && (
        <Select
          label="Line detector"
          value={value.line_detector}
          onChange={(v) => set({ line_detector: v })}
          disabled={disabled}
          options={[
            { value: 'auto', label: 'Automatic — docTR if installed, otherwise built-in' },
            { value: 'doctr', label: 'docTR neural detector (pip install python-doctr)' },
            { value: 'projection', label: 'Built-in projection detector — no extra library' },
          ]}
        />
      )}

      <Select
        label="Decoding"
        value={value.decode_strategy}
        onChange={(v) => set({ decode_strategy: v })}
        disabled={disabled || !supportsBeam}
        options={[
          { value: 'greedy', label: 'Greedy — fastest' },
          {
            value: 'beam',
            label: supportsBeam
              ? 'Beam search — slower, slightly more accurate'
              : 'Beam search — not available for this model',
            disabled: !supportsBeam,
          },
        ]}
      />
      {!supportsBeam && (
        <p className="-mt-2 text-[11px] text-slate-500">
          Transformer OCR generates its output autoregressively, so CTC beam search does not apply.
        </p>
      )}

      {value.decode_strategy === 'beam' && supportsBeam && (
        <Slider
          label="Beam width"
          min={2}
          max={25}
          value={value.beam_width}
          onChange={(v) => set({ beam_width: v })}
        />
      )}

      <Slider
        label="Minimum line confidence"
        min={0}
        max={0.9}
        step={0.05}
        value={value.min_confidence}
        onChange={(v) => set({ min_confidence: v })}
        format={(v) => (v === 0 ? 'off' : `${(v * 100).toFixed(0)}%`)}
      />

      <Toggle
        label="Save to history"
        description="Keep this request and its image for later review"
        checked={value.save_to_history}
        onChange={(v) => set({ save_to_history: v })}
        disabled={disabled}
      />
    </div>
  )
}
