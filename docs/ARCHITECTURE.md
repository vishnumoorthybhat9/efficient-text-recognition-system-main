# Architecture

How the system is put together, and why each significant decision was made.

---

## 1. Layering

```
┌─────────────────────────────────────────────────────────┐
│ frontend/      React 18 + Vite + Tailwind v4            │
│                talks only to /api over HTTP             │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│ backend/       FastAPI                                  │
│   api/routes   HTTP contracts, validation               │
│   services/    use cases; the only layer that knows     │
│                about both the database and ml/          │
│   db/          SQLAlchemy models and sessions           │
│   core/        settings, error handling                 │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│ ml/            standalone deep-learning library         │
│                imports nothing from backend/            │
└─────────────────────────────────────────────────────────┘
```

**`ml/` never imports `backend/`.** The library stands alone: the training
scripts use it with no web server involved, and the API is simply one consumer.
This is what lets `scripts/train_crnn.py` run on a machine that has never
started the server.

Routes contain no business logic — they validate input, call a service and
shape the response. Services contain no HTTP concepts. That separation is why
`scripts/evaluate.py` can reuse `backend.services.evaluation_service` directly.

---

## 2. Request flow

```
upload ─► storage.read_upload       validate size/format, decode, store
       ─► PreprocessPipeline        OpenCV enhancement
       ─► detect_text_lines         projection-profile line segmentation
       ─► predictor.recognize_lines batched model inference
       ─► ctc decode                greedy or prefix beam search
       ─► OCRRequest row            persisted to SQLite
       ─► RecognitionResponse       text, lines, confidence, timings
```

Steps 2–3 are shared by every model. Only step 4 differs between the CRNN
recogniser and TrOCR, which is what makes the model comparison fair — and why
the timing breakdown is reported per stage rather than as one number.

---

## 3. The recogniser

```
image (B, 1, 32, 256)
  → CNNEncoder                 → (B, 63, 512)
  → BiLSTM × 2 (256/direction) → (B, 63, 512)
  → Self-attention (8 heads)   → (B, 63, 512)
  → Linear                     → (B, 63, 84)
  → CTC
```

### Geometry

Height 32, width 256. The encoder downsamples width by 4 and its final valid
2×2 convolution trims one column, giving **T = 256/4 − 1 = 63** time steps.
CTC needs a blank between repeated characters, so the usable label length is
capped at ~37 characters; `validate_manifest` enforces this before training
rather than letting it surface as an `inf` loss.

Images are scaled by height and right-padded, never stretched — distorting
glyph aspect ratio measurably hurts CER. The fraction of the canvas holding
real content is carried through as `width_ratio` so the BiLSTM can be packed
and the attention masked, instead of both processing padding.

### Why self-attention, not an attention decoder

This is the one architectural decision most likely to be questioned.

A Bahdanau-style attention decoder emits its own output sequence whose length
is unrelated to the input width. CTC requires one output frame per input frame
in monotonic order. The two are incompatible; using a decoder would mean
dropping CTC for cross-entropy with teacher forcing, which is a different
system from the one the synopsis specifies.

Placing multi-head self-attention **between** the BiLSTM and the CTC
classifier keeps `T` unchanged while giving every frame a global receptive
field — so the model can use evidence from anywhere in the line to disambiguate
one character, which a BiLSTM alone does poorly over long spans.

`tests/test_models.py::test_attention_is_length_preserving` pins this.

### Implementation details that matter

- **Pre-norm transformer blocks.** Trains stably without a depth-tuned warm-up
  schedule, which matters for a few dozen epochs on a laptop GPU.
- **Fixed sinusoidal positional encoding.** Self-attention is permutation
  invariant; without position the model loses character order. Fixed rather
  than learned so it generalises to unseen line widths.
- **Forget-gate bias initialised to 1.** Shortens the early-epoch CTC plateau.
- **`log_probs` forced to float32.** Under bfloat16 autocast, CTC loss is
  numerically unstable. The model casts before the log-softmax.
- **Packed sequences.** Without them the backward LSTM pass starts inside the
  right-hand padding.

---

## 4. Self-supervised pretraining

`SimCLRModel` wraps **the same `CNNEncoder` class** the recogniser uses, so the
transfer is an exact `state_dict` load rather than a hand-written key mapping.
`CRNNAttentionOCR.load_encoder_weights` returns an `EncoderTransferReport`
stating how many tensors actually moved, and that report is written into the
training history.

### OCR-aware augmentation

The canonical SimCLR recipe — random resized crop, colour jitter, horizontal
flip — is wrong for text. Flipping a word changes what it says; aggressive
cropping removes characters. Both would teach invariance to the very signal
the downstream task needs.

What is used instead: blur, Gaussian noise, stroke erosion/dilation, JPEG
artefacts, slant/shear, brightness and contrast, and width cropping bounded at
90 %. These are nuisances a scanner or camera genuinely introduces.

`horizontal_flip` is not merely defaulted off — `AugmentOptions(horizontal_flip=True)`
raises, so the mistake cannot be made quietly.

### The projection head

Discarded after pretraining. The contrastive objective is invariant to the
augmentations by construction; acting directly on the encoder output would
strip information the recogniser still needs (stroke shape, position). The head
absorbs that invariance so the encoder does not have to —
`SimCLRModel.encoder_state_dict()` returns the encoder alone.

### Both views in one forward pass

The projection head contains BatchNorm. Running the two views as separate
batches would normalise each with its own statistics, leaking the pair
structure into the features and letting the model partly cheat the objective.
They are concatenated into one batch.

---

## 5. Line segmentation

Classical morphology, not a learned detector. The synopsis is about
*recognition*; a morphological line finder is reliable on documents, needs no
training data and costs a few milliseconds.

The pipeline estimates a median character height from connected components,
sizes every subsequent kernel relative to it (so one parameter set works on a
200 px crop and a 3000 px scan), dilates horizontally to merge characters into
lines, then takes connected components as candidate lines.

**The projection profile is the authority on line count.** A local threshold
such as Sauvola shatters glyphs into stroke fragments, which collapses the
median component height and made a single cropped line look like several. A
row-projection profile is immune: a text line is one contiguous run of inked
rows however fragmented its strokes. This was a real bug, found by
`tests/test_segmentation.py`, and the fix is why that test is parametrised over
every binarisation mode.

---

## 6. Data flow and the manifest format

Every dataset — synthetic, Kaggle handwriting, IAM, IIIT5K — converts to one
two-column TSV:

```
images/train/000001.png<TAB>Invoice total
```

Paths are relative to the manifest, so a prepared dataset is movable between
machines. One `Dataset` implementation serves all sources, which means
switching the whole experiment to a different corpus is a config path change.

Importers report exactly what they discarded and why. On the Kaggle handwriting
set that meant 888 rows the annotators marked `UNREADABLE` — training on those
would teach the model to emit that literal word.

---

## 7. Model registry

Discovery reads only the small `history.json` beside each checkpoint; weights
load on first use. On a 4 GB GPU this is not an optimisation but a requirement
— loading every model at start-up would not fit.

Every entry carries **provenance**:

- `trained_in_project` — its metrics are results of this work.
- `pretrained_third_party` — downloaded weights, a baseline only.

The UI badges this everywhere a model appears, so a downloaded model's numbers
can never be mistaken for a project result.

---

## 8. Persistence

SQLite via SQLAlchemy 2.0, with WAL journal mode so a long training job writing
progress does not block reads. Foreign keys are enabled per connection (SQLite
disables them by default, which would make `ondelete="CASCADE"` silently do
nothing).

Schema is created with `create_all` rather than migrations: it ships with the
application and there is no deployed database to migrate. Changing a column
means deleting `artifacts/ocr.db`, which the README states.

JSON columns hold structured detail (per-line results, preprocessing options,
per-epoch metrics). They are read back whole and never queried into, which is
what makes that acceptable.

---

## 9. Training jobs

One job at a time, in a background thread.

**A thread, not a subprocess**, because the trainers already accept a
`should_stop` callback: cancellation is cooperative, so a cancelled run saves
its checkpoint and history before returning. PyTorch releases the GIL inside
its kernels, so the API stays responsive. A subprocess would need IPC for
progress and still could not be killed safely mid-checkpoint-write.

**One at a time**, because a second concurrent run on a 4 GB GPU fails with an
out-of-memory error partway through, after wasting minutes. The manager refuses
and says why.

**Nothing trains at start-up.** A job exists only because someone asked for one.

---

## 10. Error handling

Every error returns the same envelope:

```json
{"error": "model_unavailable", "message": "…", "detail": {}, "request_id": "…"}
```

`message` is written for a person and displayed verbatim by the frontend, so it
states both what went wrong and what to do about it. 5xx responses carry a
request id that matches a full traceback in `artifacts/logs/api.log`; the
traceback text itself is never returned, since it can contain filesystem paths.

---

## 11. Measurement discipline

Anything that reports a number follows the same rules:

- **CUDA is synchronised** around every timed region. Without it the clock
  measures how long it took to *enqueue* work, not to do it.
- **Warm-up batches are discarded.** The first CUDA forward pass pays for
  kernel autotuning and allocator setup, often 10–100× steady state.
- **Image loading is timed separately** from inference — it is a cost every
  model shares, and folding it in flatters a slow model.
- **`synchronize(None)` is a no-op.** Timing a CPU stage must not pay for a
  device sync. Before this was made explicit, the preprocessing pipeline spent
  22.4 ms per image synchronising CUDA to time OpenCV calls that took 0.13 ms —
  a 168× overhead on the hot path.
