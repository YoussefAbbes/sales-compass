import { useState } from 'react'
import { Link } from 'react-router-dom'
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import {
  endpoints,
  formatMoney,
  formatNumber,
  formatPercent,
  type Vendor,
  type VendorDashboard,
} from '../api'
import {
  BrandPortfolio,
  chartTooltipStyle,
  ErrorState,
  Loading,
  MetricCard,
  NoticePanel,
  PageHeader,
  RankBadge,
  SectionTitle,
  StatusPill,
} from '../components/ui'
import { useFetch } from '../useFetch'

export default function VendorsPage() {
  const vendors = useFetch<{ items: Vendor[] }>(endpoints.vendors)
  const [vendorCode, setVendorCode] = useState<string | null>(null)
  const activeVendor = vendorCode ?? vendors.data?.items[0]?.vendor_code ?? null
  const dashboard = useFetch<VendorDashboard>(
    activeVendor ? endpoints.vendorDashboard(activeVendor) : null,
  )

  if (vendors.error) return <ErrorState error={vendors.error} />
  if (vendors.loading || !vendors.data) return <Loading label="Chargement des vendeurs" />

  return (
    <>
      <PageHeader
        eyebrow="Portefeuilles vendeurs"
        title="Un vendeur, son activité, ses marques."
        description="Une vue dédiée au pilotage du vendeur : taille de la file, montants, rythme commercial, marques vendues et clients prioritaires."
        action={
          <label className="select-control">
            <span>Vendeur analysé</span>
            <select value={activeVendor ?? ''} onChange={event => setVendorCode(event.target.value)}>
              {vendors.data.items.map(vendor => (
                <option key={vendor.vendor_code} value={vendor.vendor_code}>
                  {vendor.vendor_name} · {vendor.queued_clients} clients
                </option>
              ))}
            </select>
          </label>
        }
      />

      {dashboard.error ? <ErrorState error={dashboard.error} /> : null}
      {dashboard.loading || !dashboard.data ? <Loading label="Analyse du portefeuille vendeur" /> : (
        <>
          <div className="metric-grid vendor-metrics">
            <MetricCard label="Clients à traiter" value={formatNumber(dashboard.data.queued_clients)} detail="classés dans la file" tone="pine" />
            <MetricCard label="Montant de priorité" value={formatMoney(dashboard.data.tnd_at_stake, true)} detail="indice composite de classement" tone="clay" />
            <MetricCard label="Valeur attendue cumulée" value={formatMoney(dashboard.data.expected_value_tnd, true)} detail="score brut × CA passé" />
            <MetricCard label="Baisse historique cumulée" value={formatMoney(dashboard.data.lost_revenue_tnd, true)} detail="constat, pas prédiction" tone="gold" />
          </div>

          <section className="panel vendor-brand-panel">
            <div className="panel-heading-action">
              <SectionTitle
                index="01"
                title={`Marques vendues par ${dashboard.data.vendor_name}`}
                description="Le portefeuille réel du vendeur : CA positif, clients servis, références actives et évolution comparable."
              />
              <div className="portfolio-health">
                <span><strong>{formatPercent(dashboard.data.known_brand_ca_share)}</strong> CA couvert</span>
                <span><strong>{formatPercent(dashboard.data.brand_concentration_top5)}</strong> dans le top 5</span>
              </div>
            </div>
            <BrandPortfolio items={dashboard.data.brands.slice(0, 8)} />
          </section>

          <div className="dashboard-grid vendor-performance-grid">
            <section className="panel chart-panel">
              <SectionTitle
                index="02"
                title="Activité mensuelle du vendeur"
                description="CA observé sur son portefeuille, sans interpolation des mois absents."
              />
              <div className="chart-wrap vendor-activity-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={dashboard.data.monthly_ca} margin={{ top: 12, right: 16, left: 4, bottom: 4 }}>
                    <defs>
                      <linearGradient id="vendorFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#0f5c4a" stopOpacity={0.3} />
                        <stop offset="100%" stopColor="#0f5c4a" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#ebe5db" vertical={false} />
                    <XAxis dataKey="month" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} />
                    <YAxis
                      tickFormatter={(value: number) => formatMoney(value, true).replace(' TND', '')}
                      tickLine={false}
                      axisLine={false}
                      width={58}
                      tick={{ fontSize: 10 }}
                    />
                    <Tooltip formatter={(value: number) => [formatMoney(value), 'CA vendeur']} contentStyle={chartTooltipStyle} />
                    <Area type="monotone" dataKey="ca_tnd" stroke="#0f5c4a" strokeWidth={3} fill="url(#vendorFill)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="panel vendor-status-panel">
              <SectionTitle index="03" title="État de la file clients" description="Répartition par statut des clients de ce vendeur." />
              <div className="status-stack">
                {dashboard.data.status_breakdown.map(item => (
                  <div key={item.status}>
                    <StatusPill status={item.status} />
                    <strong>{formatNumber(item.count)}</strong>
                    <span>{formatMoney(item.tnd_at_stake, true)} en priorité cumulée</span>
                  </div>
                ))}
              </div>
              <div className="vendor-action-stack">
                <Link className="primary-button vendor-clients-button" to={`/experience-vendeur?vendor=${encodeURIComponent(dashboard.data.vendor_code)}`}>
                  Tester comme ce vendeur
                </Link>
                <Link className="secondary-button" to={`/clients?vendor=${encodeURIComponent(dashboard.data.vendor_code)}`}>
                  Ouvrir ses dossiers clients
                </Link>
              </div>
            </section>
          </div>

          <SectionTitle
            index="04"
            title="Clients prioritaires de ce vendeur"
            description="Une synthèse de pilotage. Le dossier complet reste dans l’espace Clients."
          />
          <div className="vendor-client-list">
            {dashboard.data.clients.slice(0, 12).map(client => (
              <article key={client.client_id}>
                <RankBadge rank={client.rank} />
                <div className="vendor-client-person">
                  <strong>{client.client_name}</strong>
                  <span className="mono">{client.client_id}</span>
                </div>
                <StatusPill status={client.status} />
                <div className="vendor-client-reason">
                  <span>Signal principal</span>
                  <p>{client.summary}</p>
                </div>
                <div className="vendor-client-value">
                  <strong>{formatMoney(client.key_amount_tnd, true)}</strong>
                  <span>{client.amount_explanation}</span>
                </div>
                <Link
                  className="row-action"
                  to={`/clients?vendor=${encodeURIComponent(activeVendor ?? '')}&client=${encodeURIComponent(client.client_id)}`}
                >
                  Voir le dossier →
                </Link>
              </article>
            ))}
          </div>

          <NoticePanel title="Lecture du portefeuille vendeur" items={dashboard.data.notices} />
        </>
      )}
    </>
  )
}
