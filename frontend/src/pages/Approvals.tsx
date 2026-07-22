import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { api } from '../api/client'
import type { PendingApproval, ApprovalDetail } from '../api/types'
import StatusBadge from '../components/StatusBadge'
import PipelineStage from '../components/PipelineStage'
import Spinner from '../components/Spinner'
import { formatDateTime } from '../lib/time'

const formatTime = (iso: string) => formatDateTime(iso, { seconds: true })

function expiresIn(iso: string): string {
  const diff = new Date(iso).getTime() - Date.now()
  if (diff <= 0) return 'Expired'
  const mins = Math.floor(diff / 60000)
  const secs = Math.floor((diff % 60000) / 1000)
  return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`
}

function ApprovalCard({ approval }: { approval: PendingApproval }) {
  const qc = useQueryClient()
  const [expanded, setExpanded] = useState(false)
  const [rejectReason, setRejectReason] = useState('')
  const [showRejectInput, setShowRejectInput] = useState(false)
  const [confirmApprove, setConfirmApprove] = useState(false)

  const { data: detail, isLoading: detailLoading } = useQuery<ApprovalDetail>({
    queryKey: ['approval-detail', approval.approval_id],
    queryFn: () => api.getApprovalDetail(approval.approval_id),
    enabled: expanded,
  })

  const approveMut = useMutation({
    mutationFn: () => api.approveOrder(approval.approval_id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['live-approvals'] })
      setConfirmApprove(false)
    },
  })

  const rejectMut = useMutation({
    mutationFn: () => api.rejectOrder(approval.approval_id, rejectReason),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['live-approvals'] })
      setShowRejectInput(false)
      setRejectReason('')
    },
  })

  const isPending = approval.status === 'pending'
  const topSymbols = approval.orders.slice(0, 3).map((o) => o.symbol).join(', ')
  const moreCount = approval.orders.length - 3

  return (
    <div className="bg-slate-800 rounded-xl border border-slate-700 overflow-hidden">
      {/* Card header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-5 py-4 flex items-center justify-between hover:bg-slate-700/30 transition-colors text-left"
      >
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-1">
            <StatusBadge status={approval.status} />
            <span className="text-xs text-slate-400">{formatTime(approval.created_at)}</span>
            {isPending && (
              <span className="text-xs text-amber-400">
                Expires in {expiresIn(approval.expires_at)}
              </span>
            )}
          </div>
          <p className="text-sm text-slate-300">
            {approval.orders.length} trade{approval.orders.length !== 1 ? 's' : ''}
            {topSymbols && (
              <span className="text-slate-400"> — {topSymbols}{moreCount > 0 ? ` +${moreCount} more` : ''}</span>
            )}
          </p>
        </div>
        <span className="text-slate-500 text-xs ml-4">{expanded ? '▲' : '▼'}</span>
      </button>

      {/* Expanded detail */}
      {expanded && (
        <div className="border-t border-slate-700 px-5 py-4 space-y-4">
          {/* Orders table */}
          <div>
            <h4 className="text-xs font-semibold text-slate-400 mb-2 uppercase">Proposed Trades</h4>
            <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-left text-slate-500">
                  <th className="pr-4 pb-1">Symbol</th>
                  <th className="pr-4 pb-1">Side</th>
                  <th className="pr-4 pb-1 text-right">Qty</th>
                  <th className="pr-4 pb-1">Type</th>
                  <th className="pr-4 pb-1">Strategy</th>
                </tr>
              </thead>
              <tbody>
                {approval.orders.map((o, i) => (
                  <tr key={i} className="text-slate-300">
                    <td className="pr-4 py-0.5 font-mono">{o.symbol}</td>
                    <td className={`pr-4 py-0.5 ${o.side === 'buy' ? 'text-emerald-400' : 'text-red-400'}`}>
                      {o.side}
                    </td>
                    <td className="pr-4 py-0.5 text-right font-mono">{o.quantity}</td>
                    <td className="pr-4 py-0.5">{o.order_type}</td>
                    <td className="pr-4 py-0.5">{o.strategy}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            </div>
          </div>

          {/* Pipeline context */}
          {detailLoading && (
            <div className="flex justify-center py-4">
              <Spinner className="h-5 w-5" />
            </div>
          )}

          {detail?.blackboard_snapshot && (
            <div className="space-y-3">
              {/* Proposal */}
              {detail.blackboard_snapshot.proposal && (
                <PipelineStage title="PM Rationale" status="complete">
                  <div className="text-xs text-slate-300">
                    {detail.blackboard_snapshot.proposal.notes && <p className="italic mb-2">{detail.blackboard_snapshot.proposal.notes}</p>}
                    <div className="flex flex-wrap gap-3">
                      {Object.entries(detail.blackboard_snapshot.proposal.targets).map(([sym, wt]) => (
                        <span key={sym} className="font-mono">
                          {sym}: {(wt * 100).toFixed(1)}%
                        </span>
                      ))}
                    </div>
                  </div>
                </PipelineStage>
              )}

              {/* Risk Decision */}
              {detail.blackboard_snapshot.risk_decision && (
                <PipelineStage title="Risk Assessment" status="complete">
                  <div className="text-xs">
                    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium mb-2 ${
                      detail.blackboard_snapshot.risk_decision.approved
                        ? 'bg-emerald-900/50 text-emerald-400'
                        : 'bg-red-900/50 text-red-400'
                    }`}>
                      {detail.blackboard_snapshot.risk_decision.approved ? 'APPROVED' : 'REJECTED'}
                    </span>
                    {detail.blackboard_snapshot.risk_decision.violations.length > 0 && (
                      <ul className="list-disc list-inside text-red-400 space-y-0.5 mt-2">
                        {detail.blackboard_snapshot.risk_decision.violations.map((v, i) => (
                          <li key={i}>{v}</li>
                        ))}
                      </ul>
                    )}
                    {detail.blackboard_snapshot.risk_decision.actions.length > 0 && (
                      <ul className="list-disc list-inside text-amber-400 space-y-0.5 mt-2">
                        {detail.blackboard_snapshot.risk_decision.actions.map((a, i) => (
                          <li key={i}>{a}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                </PipelineStage>
              )}
            </div>
          )}

          {/* Action buttons */}
          {isPending && (
            <div className="flex items-center gap-3 pt-2">
              {!confirmApprove && !showRejectInput && (
                <>
                  <button
                    onClick={() => setConfirmApprove(true)}
                    className="px-4 py-2 text-sm font-medium rounded-lg bg-emerald-600 text-white hover:bg-emerald-500 transition-colors"
                  >
                    Approve
                  </button>
                  <button
                    onClick={() => setShowRejectInput(true)}
                    className="px-4 py-2 text-sm font-medium rounded-lg bg-red-600 text-white hover:bg-red-500 transition-colors"
                  >
                    Reject
                  </button>
                </>
              )}

              {confirmApprove && (
                <div className="flex items-center gap-3">
                  <span className="text-xs text-amber-400">Confirm approval?</span>
                  <button
                    onClick={() => approveMut.mutate()}
                    disabled={approveMut.isPending}
                    className="px-4 py-2 text-sm font-medium rounded-lg bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-40 transition-colors flex items-center gap-2"
                  >
                    {approveMut.isPending && <Spinner className="h-3.5 w-3.5" />}
                    Yes, Approve
                  </button>
                  <button
                    onClick={() => setConfirmApprove(false)}
                    className="px-3 py-2 text-sm text-slate-400 hover:text-slate-200 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              )}

              {showRejectInput && (
                <div className="flex items-center gap-3 flex-1">
                  <input
                    type="text"
                    value={rejectReason}
                    onChange={(e) => setRejectReason(e.target.value)}
                    placeholder="Reason for rejection..."
                    className="flex-1 px-3 py-2 bg-slate-700 border border-slate-600 rounded-lg text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-red-500"
                  />
                  <button
                    onClick={() => rejectMut.mutate()}
                    disabled={rejectMut.isPending || !rejectReason.trim()}
                    className="px-4 py-2 text-sm font-medium rounded-lg bg-red-600 text-white hover:bg-red-500 disabled:opacity-40 transition-colors flex items-center gap-2"
                  >
                    {rejectMut.isPending && <Spinner className="h-3.5 w-3.5" />}
                    Reject
                  </button>
                  <button
                    onClick={() => { setShowRejectInput(false); setRejectReason('') }}
                    className="px-3 py-2 text-sm text-slate-400 hover:text-slate-200 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>
          )}

          {approveMut.error && (
            <div className="bg-red-900/20 border border-red-700 rounded-lg p-3 text-red-400 text-sm">
              {(approveMut.error as Error).message}
            </div>
          )}
          {rejectMut.error && (
            <div className="bg-red-900/20 border border-red-700 rounded-lg p-3 text-red-400 text-sm">
              {(rejectMut.error as Error).message}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function Approvals() {
  const qc = useQueryClient()
  const [tab, setTab] = useState<'pending' | 'history'>('pending')
  const [confirmClear, setConfirmClear] = useState(false)

  const { data: approvals, isLoading, error } = useQuery<PendingApproval[]>({
    queryKey: ['live-approvals'],
    queryFn: api.getApprovals,
    refetchInterval: 3000,
  })

  const clearMut = useMutation({
    mutationFn: () => api.clearApprovals(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['live-approvals'] })
      setConfirmClear(false)
    },
  })

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Spinner className="h-8 w-8" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-700 rounded-xl p-6 text-red-400">
        <h3 className="font-semibold mb-1">Failed to load approvals</h3>
        <p className="text-sm">{(error as Error).message}</p>
      </div>
    )
  }

  const pending = approvals?.filter((a) => a.status === 'pending') ?? []
  const history = approvals?.filter((a) => a.status !== 'pending') ?? []
  const list = tab === 'pending' ? pending : history

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white">Approvals</h2>
          <p className="text-sm text-slate-400 mt-1">
            {pending.length} pending approval{pending.length !== 1 ? 's' : ''}
          </p>
        </div>
        {approvals && approvals.length > 0 && (
          confirmClear ? (
            <div className="flex items-center gap-2">
              <span className="text-xs text-amber-400">Delete all {approvals.length} approvals?</span>
              <button
                onClick={() => clearMut.mutate()}
                disabled={clearMut.isPending}
                className="px-3 py-2 text-sm font-medium rounded-lg bg-red-600 text-white hover:bg-red-500 disabled:opacity-40 transition-colors flex items-center gap-2"
              >
                {clearMut.isPending && <Spinner className="h-3.5 w-3.5" />}
                Yes, Clear All
              </button>
              <button
                onClick={() => setConfirmClear(false)}
                className="px-3 py-2 text-sm text-slate-400 hover:text-slate-200 transition-colors"
              >
                Cancel
              </button>
            </div>
          ) : (
            <button
              onClick={() => setConfirmClear(true)}
              className="px-4 py-2 text-sm font-medium rounded-lg border border-red-700 text-red-400 hover:bg-red-900/20 transition-colors"
            >
              Clear All
            </button>
          )
        )}
      </div>
      {clearMut.error && (
        <div className="bg-red-900/20 border border-red-700 rounded-xl p-4 text-red-400 mb-4 text-sm">
          {(clearMut.error as Error).message}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-2 mb-6">
        <button
          onClick={() => setTab('pending')}
          className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            tab === 'pending'
              ? 'bg-blue-600/20 text-blue-400 border border-blue-500/40'
              : 'border border-slate-600 text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          Pending ({pending.length})
        </button>
        <button
          onClick={() => setTab('history')}
          className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
            tab === 'history'
              ? 'bg-blue-600/20 text-blue-400 border border-blue-500/40'
              : 'border border-slate-600 text-slate-400 hover:text-slate-200 hover:bg-slate-800'
          }`}
        >
          History ({history.length})
        </button>
      </div>

      {/* Cards */}
      {list.length === 0 ? (
        <div className="bg-slate-800 rounded-xl border border-slate-700 p-12 text-center">
          <p className="text-slate-400">
            {tab === 'pending' ? 'No pending approvals.' : 'No approval history yet.'}
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {list.map((a) => (
            <ApprovalCard key={a.approval_id} approval={a} />
          ))}
        </div>
      )}
    </div>
  )
}
