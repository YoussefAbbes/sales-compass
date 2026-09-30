import type { ReactNode } from 'react'

import {
  formatMoney,
  formatNumber,
  formatPercent,
  statusLabel,
  type BrandSummary,
} from '../api'

export function PageHeader({ eyebrow, title, description, action }: {
  eyebrow: string
  title: string
  description: string
  action?: ReactNode
}) {
  return (
    <header className="page-header">
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      {action ? <div className="page-action">{action}</div> : null}
    </header>
  )
}

export function MetricCard({ label, value, detail, tone = 'ink' }: {
  label: string
  value: ReactNode
  detail?: string
  tone?: 'ink' | 'pine' | 'clay' | 'gold'
}) {
  return (
    <article className={`metric-card tone-${tone}`}>
      <span className="metric-label">{label}</span>
      <strong>{value}</strong>
      {detail ? <p>{detail}</p> : null}
    </article>
  )
}

export function SectionTitle({ index, title, description }: {
  index?: string
  title: string
  description?: string
}) {
  return (
    <div className="section-title">
      {index ? <span>{index}</span> : null}
      <div>
        <h2>{title}</h2>
        {description ? <p>{description}</p> : null}
      </div>
    </div>
  )
}

export function StatusPill({ status }: { status: string }) {
  const css = status === 'INACTIVE_2026' ? 'danger' : status === 'DECLINING' ? 'warning' : 'steady'
  return <span className={`status-pill ${css}`}>{statusLabel(status)}</span>
}

export function RankBadge({ rank }: { rank: number }) {
  return <span className="rank-badge">#{rank}</span>
}

export function WhyList({ items }: { items: string[] }) {
  return (
    <ul className="why-list">
      {items.map((item, index) => (
        <li key={`${index}-${item}`}><span>{index + 1}</span><p>{item}</p></li>
      ))}
    </ul>
  )
}

export function TrendPill({ value }: { value: number | null }) {
  if (value === null) return <span className="trend-pill neutral">Nouveau en 2026</span>
  const tone = value > 0.001 ? 'positive' : value < -0.001 ? 'negative' : 'neutral'
  const prefix = value > 0 ? '+' : ''
  return <span className={`trend-pill ${tone}`}>{prefix}{formatPercent(value, 0)}</span>
}

export function BrandPortfolio({ items, compact = false }: {
  items: BrandSummary[]
  compact?: boolean
}) {
  if (items.length === 0) {
    return <EmptyState title="Aucune marque identifiable">Les références de ce périmètre ne sont pas reliées à une marque connue.</EmptyState>
  }

  const maximum = Math.max(...items.map(item => item.ca_tnd), 1)
  return (
    <div className={`brand-portfolio ${compact ? 'compact' : ''}`}>
      {items.map(item => (
        <article className="brand-row" key={item.brand}>
          <div className="brand-row-head">
            <span className="brand-rank">{String(item.rank).padStart(2, '0')}</span>
            <strong>{item.brand}</strong>
            <TrendPill value={item.evolution_pct} />
          </div>
          <div className="brand-bar" aria-label={`${item.brand}, ${formatPercent(item.share_of_scope_ca)}`}>
            <i style={{ width: `${Math.max((item.ca_tnd / maximum) * 100, 2)}%` }} />
          </div>
          <div className="brand-row-stats">
            <span><strong>{formatMoney(item.ca_tnd, true)}</strong> CA</span>
            <span><strong>{formatPercent(item.share_of_scope_ca)}</strong> du portefeuille connu</span>
            <span><strong>{formatNumber(item.clients)}</strong> client{item.clients > 1 ? 's' : ''}</span>
            <span><strong>{formatNumber(item.articles)}</strong> référence{item.articles > 1 ? 's' : ''}</span>
          </div>
        </article>
      ))}
    </div>
  )
}

export function ShowMore({ shown, total, initial, step, onChange }: {
  shown: number
  total: number
  initial: number
  step: number
  onChange: (next: number) => void
}) {
  const remaining = total - shown
  // Nothing hidden and nothing expanded: the control would be an empty affordance.
  if (remaining <= 0 && shown <= initial) return null
  return (
    <div className="show-more">
      {remaining > 0 ? (
        <button type="button" onClick={() => onChange(Math.min(total, shown + step))}>
          Afficher plus <span className="show-more-count">
            ({remaining} restant{remaining > 1 ? 's' : ''})
          </span>
        </button>
      ) : null}
      {shown > initial ? (
        <button type="button" className="show-more-less" onClick={() => onChange(initial)}>
          Réduire
        </button>
      ) : null}
    </div>
  )
}


export function NoticePanel({ title = 'À garder en tête', items }: {
  title?: string
  items: string[]
}) {
  return (
    <aside className="notice-panel">
      <div className="notice-icon">!</div>
      <div>
        <strong>{title}</strong>
        <ul>{items.map(item => <li key={item}>{item}</li>)}</ul>
      </div>
    </aside>
  )
}

export function Formula({ children }: { children: ReactNode }) {
  return <div className="formula">{children}</div>
}

export function Loading({ label = 'Chargement des informations' }: { label?: string }) {
  return (
    <div className="state-card loading-state">
      <span className="loader" />
      <div><strong>{label}</strong><p>Les calculs restent sur le serveur local.</p></div>
    </div>
  )
}

export function ErrorState({ error }: { error: unknown }) {
  const message = error instanceof Error ? error.message : String(error)
  return (
    <div className="state-card error-state">
      <span>×</span>
      <div>
        <strong>Impossible de charger cette vue</strong>
        <p>{message}. Vérifiez que l’API locale tourne sur le port 8001.</p>
      </div>
    </div>
  )
}

export function EmptyState({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="state-card empty-state">
      <span>—</span>
      <div><strong>{title}</strong><p>{children}</p></div>
    </div>
  )
}

export function MoneyBreakdown({ expected, lost }: { expected: number; lost: number }) {
  return (
    <div className="money-breakdown">
      <div><span>Valeur attendue modèle</span><strong>{formatMoney(expected)}</strong></div>
      <div><span>Baisse historique constatée</span><strong>{formatMoney(lost)}</strong></div>
    </div>
  )
}

export const chartTooltipStyle = {
  background: '#fffdf9',
  border: '1px solid #e6e0d6',
  borderRadius: '12px',
  boxShadow: '0 12px 36px rgba(20,35,31,.12)',
  fontSize: '12px',
}
