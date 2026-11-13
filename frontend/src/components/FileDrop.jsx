import { useCallback, useEffect, useRef, useState } from 'react'
import { ImageIcon, Upload, X } from 'lucide-react'
import { Button } from './ui.jsx'
import { bytes, cx } from '../lib/format.js'

const ACCEPTED = ['image/png', 'image/jpeg', 'image/bmp', 'image/tiff', 'image/webp']
const ACCEPT_ATTR = '.png,.jpg,.jpeg,.bmp,.tif,.tiff,.webp'

/**
 * Drag-and-drop image picker with a preview.
 *
 * Also accepts a paste from the clipboard, which is how most people move a
 * screenshot into a browser.
 *
 * @param {File|null} file          the selected file
 * @param {Function} onChange       called with the File, or null when cleared
 * @param {number} maxSizeMb        client-side size guard; the server enforces its own
 */
export function FileDrop({ file, onChange, maxSizeMb = 15, disabled = false, className }) {
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const inputRef = useRef(null)

  // Object URLs must be revoked or the blob stays in memory for the life of
  // the page; a user trying twenty images would leak all twenty.
  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return undefined
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const accept = useCallback(
    (candidate) => {
      if (!candidate) return
      setError(null)

      if (candidate.type && !ACCEPTED.includes(candidate.type) && !candidate.type.startsWith('image/')) {
        setError(`${candidate.name} is not an image (${candidate.type || 'unknown type'}).`)
        return
      }
      if (candidate.size > maxSizeMb * 1024 * 1024) {
        setError(`${candidate.name} is ${bytes(candidate.size)}, over the ${maxSizeMb} MB limit.`)
        return
      }
      onChange(candidate)
    },
    [maxSizeMb, onChange],
  )

  useEffect(() => {
    if (disabled) return undefined
    const onPaste = (event) => {
      const item = Array.from(event.clipboardData?.items || []).find((i) =>
        i.type.startsWith('image/'),
      )
      if (item) accept(item.getAsFile())
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [accept, disabled])

  const onDrop = (event) => {
    event.preventDefault()
    setDragging(false)
    if (disabled) return
    accept(event.dataTransfer.files?.[0])
  }

  if (file && previewUrl) {
    return (
      <div className={cx('relative', className)}>
        <div className="overflow-hidden rounded-xl border border-slate-200 bg-slate-100 dark:border-slate-800 dark:bg-slate-900">
          <img
            src={previewUrl}
            alt={file.name}
            className="mx-auto max-h-72 w-auto object-contain"
          />
        </div>
        <div className="mt-2 flex items-center justify-between gap-3">
          <div className="min-w-0">
            <p className="truncate text-xs font-medium text-slate-700 dark:text-slate-300">{file.name}</p>
            <p className="text-[11px] text-slate-500">{bytes(file.size)}</p>
          </div>
          <div className="flex gap-2">
            <Button size="sm" variant="secondary" onClick={() => inputRef.current?.click()} disabled={disabled}>
              Replace
            </Button>
            <Button size="sm" variant="ghost" icon={X} onClick={() => onChange(null)} disabled={disabled}>
              Clear
            </Button>
          </div>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT_ATTR}
          className="hidden"
          onChange={(e) => accept(e.target.files?.[0])}
        />
        {error && <p className="mt-2 text-xs text-rose-600">{error}</p>}
      </div>
    )
  }

  return (
    <div className={className}>
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload an image"
        onClick={() => !disabled && inputRef.current?.click()}
        onKeyDown={(e) => {
          if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
            e.preventDefault()
            inputRef.current?.click()
          }
        }}
        onDragOver={(e) => {
          e.preventDefault()
          if (!disabled) setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={cx(
          'flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-12 text-center transition-colors',
          dragging
            ? 'border-indigo-500 bg-indigo-50 dark:bg-indigo-500/10'
            : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50 dark:border-slate-700 dark:hover:border-indigo-500 dark:hover:bg-slate-900',
          disabled && 'cursor-not-allowed opacity-50',
        )}
      >
        <span className="rounded-full bg-slate-100 p-3 text-slate-400 dark:bg-slate-800">
          {dragging ? <ImageIcon className="h-6 w-6" /> : <Upload className="h-6 w-6" />}
        </span>
        <p className="mt-3 text-sm font-medium text-slate-700 dark:text-slate-300">
          {dragging ? 'Drop the image here' : 'Drag an image here, or click to browse'}
        </p>
        <p className="mt-1 text-xs text-slate-500">
          PNG, JPEG, BMP, TIFF or WebP, up to {maxSizeMb} MB — you can also paste from the clipboard
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT_ATTR}
        className="hidden"
        disabled={disabled}
        onChange={(e) => accept(e.target.files?.[0])}
      />
      {error && <p className="mt-2 text-xs text-rose-600">{error}</p>}
    </div>
  )
}
