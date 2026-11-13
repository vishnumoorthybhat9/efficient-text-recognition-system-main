import { Suspense, lazy } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout.jsx'
import { Loading } from './components/ui.jsx'

// Each page is code-split so the initial load does not pull in Plotly or the
// evaluation tables for someone who only wants to recognise one image.
const Dashboard = lazy(() => import('./pages/Dashboard.jsx'))
const Recognize = lazy(() => import('./pages/Recognize.jsx'))
const PreprocessLab = lazy(() => import('./pages/PreprocessLab.jsx'))
const Compare = lazy(() => import('./pages/Compare.jsx'))
const Evaluation = lazy(() => import('./pages/Evaluation.jsx'))
const Training = lazy(() => import('./pages/Training.jsx'))
const Datasets = lazy(() => import('./pages/Datasets.jsx'))
const HistoryPage = lazy(() => import('./pages/History.jsx'))
const About = lazy(() => import('./pages/About.jsx'))

export default function App() {
  return (
    <Layout>
      <Suspense fallback={<Loading label="Loading page…" />}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/recognize" element={<Recognize />} />
          <Route path="/preprocess" element={<PreprocessLab />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/training" element={<Training />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/about" element={<About />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </Layout>
  )
}
