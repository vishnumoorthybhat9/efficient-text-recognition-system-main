import { BookOpen, Cpu, GitBranch, Layers, ShieldCheck } from 'lucide-react'
import { api } from '../api/client.js'
import { useAsync } from '../hooks/useAsync.js'
import { PageHeader } from '../components/Layout.jsx'
import { Badge, Card, CardHeader, Loading } from '../components/ui.jsx'
import { bytes } from '../lib/format.js'

const TEAM = [
  ['Vishnumoorthy Raghavendra Bhat', 'NNM23IS215'],
  ['Tejaswi Shankar P', 'NNM23IS194'],
  ['Pranava Kumara K', 'NNM23IS132'],
  ['Simran Kasim Sayed', 'NNM23IS179'],
]

const PIPELINE = [
  {
    stage: '1 · Self-supervised pretraining',
    detail:
      'SimCLR trains the CNN encoder on unlabeled text crops with the NT-Xent contrastive loss. ' +
      'No transcription is read. The augmentations are OCR-specific — blur, noise, stroke ' +
      'erosion/dilation, slant, JPEG artefacts — and deliberately exclude horizontal flipping, ' +
      'which would invalidate the label.',
  },
  {
    stage: '2 · Supervised recognition',
    detail:
      'The same encoder feeds a two-layer BiLSTM, a multi-head self-attention block and a CTC ' +
      'classifier. Self-attention sits between the BiLSTM and CTC so the sequence length is ' +
      'preserved — an attention decoder would emit its own length and break CTC’s monotonic ' +
      'alignment.',
  },
  {
    stage: '3 · Controlled comparison',
    detail:
      'Two models are trained on an identical labeled subset with the same seed, augmentation, ' +
      'optimiser and schedule. They differ in exactly one thing: whether the encoder starts from ' +
      'the SimCLR weights. The CER difference is therefore attributable to the pretraining.',
  },
  {
    stage: '4 · Transformer baseline',
    detail:
      'TrOCR (ViT encoder + text decoder) is included for reference. Its weights are pretrained ' +
      'by Microsoft and downloaded on first use; it is labelled as an external baseline and is ' +
      'not part of the self-supervised experiment.',
  },
]

export default function About() {
  const system = useAsync(() => api.systemInfo(), [])
  const models = useAsync(() => api.models(), [])

  const gpu = system.data?.gpus?.[0]

  return (
    <div>
      <PageHeader
        title="About this project"
        description="Efficient Text Recognition from Images Using Deep Learning and Self-Supervised Learning — B.Tech Major Project, Project Code 42."
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="space-y-6">
          <Card>
            <CardHeader title="What the system does" icon={BookOpen} />
            <div className="space-y-4 p-5 text-[13px] leading-relaxed text-slate-600 dark:text-slate-400">
              <p>
                Deep OCR works well but needs a great deal of labeled data, which is expensive for
                handwriting in particular. This project tests whether{' '}
                <strong className="text-slate-900 dark:text-slate-200">
                  self-supervised pretraining on unlabeled images
                </strong>{' '}
                can recover most of that accuracy when labels are scarce.
              </p>
              <p>
                The method is a controlled experiment rather than a demonstration: one recogniser
                architecture is trained twice on the same small labeled subset, differing only in
                whether its CNN encoder was first pretrained with SimCLR on unlabeled crops. The
                Evaluation page reports the result, measured on a held-out split.
              </p>
            </div>
          </Card>

          <Card>
            <CardHeader title="Pipeline" icon={GitBranch} />
            <div className="divide-y divide-slate-100 dark:divide-slate-800">
              {PIPELINE.map((item) => (
                <div key={item.stage} className="px-5 py-4">
                  <h3 className="text-sm font-semibold">{item.stage}</h3>
                  <p className="mt-1 text-[13px] leading-relaxed text-slate-600 dark:text-slate-400">
                    {item.detail}
                  </p>
                </div>
              ))}
            </div>
          </Card>

          <Card>
            <CardHeader title="How results are reported" icon={ShieldCheck} />
            <ul className="space-y-2.5 p-5 text-[13px] leading-relaxed text-slate-600 dark:text-slate-400">
              <li>
                <strong className="text-slate-900 dark:text-slate-200">Every number is measured.</strong>{' '}
                Accuracy figures come from running a model over a split; latency is wall-clock with
                CUDA synchronisation and warm-up batches discarded.
              </li>
              <li>
                <strong className="text-slate-900 dark:text-slate-200">Provenance is always shown.</strong>{' '}
                Models trained in this project and third-party pretrained weights are badged
                differently everywhere they appear.
              </li>
              <li>
                <strong className="text-slate-900 dark:text-slate-200">
                  CER and WER are micro-averaged
                </strong>{' '}
                (total edits ÷ total reference characters or words). Error rates can exceed 1.0,
                because insertions are not bounded by the reference length; only accuracy is
                clamped.
              </li>
              <li>
                <strong className="text-slate-900 dark:text-slate-200">
                  Comparisons require matching conditions.
                </strong>{' '}
                The self-supervised verdict is reported only when both arms were evaluated on the
                same dataset and split.
              </li>
            </ul>
          </Card>
        </div>

        <div className="space-y-6">
          <Card>
            <CardHeader title="Team" />
            <div className="divide-y divide-slate-100 dark:divide-slate-800">
              {TEAM.map(([name, usn]) => (
                <div key={usn} className="flex items-center justify-between px-5 py-2.5">
                  <span className="text-sm">{name}</span>
                  <span className="nums font-mono text-[11px] text-slate-500">{usn}</span>
                </div>
              ))}
            </div>
            <div className="border-t border-slate-200 px-5 py-3 dark:border-slate-800">
              <p className="text-[11px] text-slate-500">
                Guide: Dr. Rashmi Naveen
                <br />
                Department of Information Science &amp; Engineering
              </p>
            </div>
          </Card>

          <Card>
            <CardHeader title="This machine" icon={Cpu} />
            {system.loading ? (
              <Loading />
            ) : (
              <dl className="space-y-2.5 p-5 text-xs">
                <Row label="Compute device" value={system.data?.device} />
                {gpu && (
                  <>
                    <Row label="GPU" value={gpu.name} />
                    <Row label="GPU memory" value={bytes(gpu.total_memory_mb * 1024 * 1024)} />
                    <Row label="Compute capability" value={gpu.capability} />
                  </>
                )}
                <Row label="CPU" value={system.data?.cpu_name} />
                <Row label="Cores" value={system.data?.cpu_count} />
                <Row label="System memory" value={bytes((system.data?.total_ram_mb || 0) * 1024 * 1024)} />
                <Row label="PyTorch" value={system.data?.torch_version} />
                <Row label="CUDA" value={system.data?.cuda_version || 'not available'} />
                <Row label="Platform" value={system.data?.platform} />
                <Row label="Mixed precision" value={system.data?.amp_supported ? 'enabled' : 'CPU — disabled'} />
              </dl>
            )}
          </Card>

          <Card>
            <CardHeader title="Models on this machine" icon={Layers} />
            {models.loading ? (
              <Loading />
            ) : (
              <div className="divide-y divide-slate-100 dark:divide-slate-800">
                {models.data?.models?.map((model) => (
                  <div key={model.id} className="px-5 py-3">
                    <div className="flex items-start justify-between gap-2">
                      <p className="min-w-0 truncate text-sm font-medium">{model.name}</p>
                      <Badge tone={model.provenance === 'trained_in_project' ? 'emerald' : 'amber'}>
                        {model.provenance === 'trained_in_project' ? 'Ours' : 'External'}
                      </Badge>
                    </div>
                    <p className="mt-0.5 font-mono text-[10px] text-slate-500">{model.id}</p>
                  </div>
                ))}
              </div>
            )}
          </Card>

          <Card>
            <CardHeader title="Key references" />
            <ul className="space-y-2 p-5 text-[11px] leading-relaxed text-slate-500">
              <li>Shi, Bai &amp; Yao — “An End-to-End Trainable Neural Network for Image-Based Sequence Recognition” (CRNN), 2015.</li>
              <li>Graves et al. — “Connectionist Temporal Classification”, ICML 2006.</li>
              <li>Chen et al. — “A Simple Framework for Contrastive Learning of Visual Representations” (SimCLR), ICML 2020.</li>
              <li>Li et al. — “TrOCR: Transformer-Based Optical Character Recognition with Pre-Trained Models”, arXiv:2109.10282.</li>
              <li>Vaswani et al. — “Attention Is All You Need”, NeurIPS 2017.</li>
              <li>Penarrubia, Valero-Mas &amp; Calvo-Zaragoza — “Self-Supervised Learning for Text Recognition: A Critical Survey”, IJCV 2025.</li>
            </ul>
          </Card>
        </div>
      </div>
    </div>
  )
}

function Row({ label, value }) {
  return (
    <div className="flex items-start justify-between gap-3">
      <dt className="shrink-0 text-slate-500">{label}</dt>
      <dd className="min-w-0 truncate text-right font-medium">{value ?? '—'}</dd>
    </div>
  )
}
