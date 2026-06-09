import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import NewBacktest from './pages/NewBacktest'
import RunDetail from './pages/RunDetail'
import Compare from './pages/Compare'
import AgentInspector from './pages/AgentInspector'
import LiveDashboard from './pages/LiveDashboard'
import LiveConfig from './pages/LiveConfig'
import Approvals from './pages/Approvals'
import OrderHistory from './pages/OrderHistory'

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
      </Route>
    </Routes>
  )
}
