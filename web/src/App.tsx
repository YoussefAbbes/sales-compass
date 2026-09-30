import { lazy, Suspense } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'

import Layout from './components/Layout'
import { Loading } from './components/ui'

const ArticlesPage = lazy(() => import('./pages/ArticlesPage'))
const BrandsPage = lazy(() => import('./pages/BrandsPage'))
const ClientsPage = lazy(() => import('./pages/ClientsPage'))
const DataPage = lazy(() => import('./pages/DataPage'))
const ModelsPage = lazy(() => import('./pages/ModelsPage'))
const OverviewPage = lazy(() => import('./pages/OverviewPage'))
const SellerExperiencePage = lazy(() => import('./pages/SellerExperiencePage'))
const VendorsPage = lazy(() => import('./pages/VendorsPage'))

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Suspense fallback={<Loading />}><OverviewPage /></Suspense>} />
        <Route path="vendeurs" element={<Suspense fallback={<Loading />}><VendorsPage /></Suspense>} />
        <Route path="experience-vendeur" element={<Suspense fallback={<Loading />}><SellerExperiencePage /></Suspense>} />
        <Route path="clients" element={<Suspense fallback={<Loading />}><ClientsPage /></Suspense>} />
        <Route path="marques" element={<Suspense fallback={<Loading />}><BrandsPage /></Suspense>} />
        <Route path="articles" element={<Suspense fallback={<Loading />}><ArticlesPage /></Suspense>} />
        <Route path="modeles" element={<Suspense fallback={<Loading />}><ModelsPage /></Suspense>} />
        <Route path="donnees" element={<Suspense fallback={<Loading />}><DataPage /></Suspense>} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  )
}
