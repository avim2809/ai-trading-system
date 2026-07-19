import { lazy } from 'react'
import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'

// Lazy-loaded route pages so each becomes its own chunk, keeping the initial
// bundle small (the shell/Layout loads eagerly).
const Dashboard = lazy(() => import('./pages/Dashboard'))
const NewBacktest = lazy(() => import('./pages/NewBacktest'))
const RunDetail = lazy(() => import('./pages/RunDetail'))
const Compare = lazy(() => import('./pages/Compare'))
const AgentInspector = lazy(() => import('./pages/AgentInspector'))
const LiveDashboard = lazy(() => import('./pages/LiveDashboard'))
const LiveConfig = lazy(() => import('./pages/LiveConfig'))
const Approvals = lazy(() => import('./pages/Approvals'))
const OrderHistory = lazy(() => import('./pages/OrderHistory'))
const Logs = lazy(() => import('./pages/Logs'))
const Decisions = lazy(() => import('./pages/Decisions'))

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="new" element={<NewBacktest />} />
        <Route path="runs/:runId" element={<RunDetail />} />
        <Route path="compare" element={<Compare />} />
        <Route path="inspector" element={<AgentInspector />} />
        <Route path="live" element={<LiveDashboard />} />
        <Route path="live/config" element={<LiveConfig />} />
        <Route path="live/approvals" element={<Approvals />} />
        <Route path="live/orders" element={<OrderHistory />} />
        <Route path="logs" element={<Logs />} />
        <Route path="decisions" element={<Decisions />} />
      </Route>
    </Routes>
  )
}
