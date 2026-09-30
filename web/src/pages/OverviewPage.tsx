import { Link } from 'react-router-dom'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
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
  type Overview,
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

export default function OverviewPage() {
  const { data, error, loading } = useFetch<Overview>(endpoints.overview)

  if (error) return <ErrorState error={error} />
  if (loading || !data) return <Loading label="Préparation de la vue d’ensemble" />

  const vendorChart = data.vendor_comparison.slice(0, 10).map(vendor => ({
    name: vendor.vendor_name || vendor.vendor_code,
    clients: vendor.queued_clients,
    stake: vendor.tnd_at_stake,
  }))

  return (
    <>
      <PageHeader
        eyebrow="Pilotage commercial"
        title="Décider qui appeler, avec les preuves."
        description={`Une vue consolidée de ${data.data_window}. Chaque chiffre ci-dessous peut être retracé jusqu’à l’historique mensuel et aux signaux du modèle.`}
        action={<span className="live-badge"><i /> Données locales prêtes</span>}
      />

      <div className="metric-grid hero-metrics">
        <MetricCard
          label="Clients-vendeurs prioritaires"
          value={formatNumber(data.queued_assignments)}
          detail="affectations classées dans la file d’appels"
          tone="pine"
        />
        <MetricCard
          label="Montant utilisé pour prioriser"
          value={formatMoney(data.tnd_at_stake, true)}
          detail="indice d’ordre, pas une prévision de vente"
          tone="clay"
        />
        <MetricCard
          label="Chiffre d’affaires observé"
          value={formatMoney(data.total_ca_tnd, true)}
          detail={`${formatNumber(data.fact_rows)} lignes mensuelles`}
        />
        <MetricCard
          label="Périmètre"
          value={`${formatNumber(data.vendors)} vendeurs`}
          detail={`${formatNumber(data.known_clients)} clients · ${formatNumber(data.catalog_articles)} articles`}
          tone="gold"
        />
      </div>

      <div className="dashboard-grid overview-grid">
        <section className="panel chart-panel wide-panel">
          <SectionTitle
            index="01"
            title="Activité mensuelle observée"
            description="Le chiffre d’affaires réel, sans interpolation des mois absents."
          />
          <div className="chart-wrap tall">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.monthly_ca} margin={{ top: 12, right: 16, left: 4, bottom: 4 }}>
                <CartesianGrid stroke="#ebe5db" vertical={false} />
                <XAxis dataKey="month" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
                <YAxis
                  tickFormatter={(value: number) => formatMoney(value, true).replace(' TND', '')}
                  tickLine={false}
                  axisLine={false}
                  width={58}
                  tick={{ fontSize: 11 }}
                />
                <Tooltip
                  formatter={(value: number) => [formatMoney(value), 'CA']}
                  contentStyle={chartTooltipStyle}
                />
                <Line
                  type="monotone"
                  dataKey="ca_tnd"
                  stroke="#0f5c4a"
                  strokeWidth={3}
                  dot={{ r: 3, fill: '#0f5c4a' }}
                  activeDot={{ r: 6, fill: '#e4572e' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="chart-caption">
            Le registre couvre chaque mois sans interruption. Un mois sans achat reste absent plutôt que ramené à zéro, et le dernier mois peut être partiel.
          </p>
        </section>

        <section className="panel model-snapshot">
          <SectionTitle index="02" title="Deux modèles, deux questions" />
          {data.models.map(model => (
            <article className="mini-model" key={model.name}>
              <div>
                <span>{model.name === 'repurchase' ? 'Rachat' : 'Nouveau prospect'}</span>
                <strong>{formatPercent(model.headline_value, model.name === 'repurchase' ? 1 : 1)}</strong>
              </div>
              <p>{model.headline_metric}</p>
              <small>
                contre {formatPercent(model.comparison_value)} · {model.comparison_label.toLowerCase()}
              </small>
            </article>
          ))}
          <Link className="text-link" to="/modeles">Comprendre les scores et leurs limites →</Link>
        </section>
      </div>

      <section className="panel brand-overview-panel">
        <div className="panel-heading-action">
          <SectionTitle
            index="03"
            title="Les marques qui structurent le portefeuille"
            description="Poids du CA, diversité clients et références, puis évolution 2026 versus 2025 à périmètre de mois identique."
          />
          <Link className="text-link" to="/marques">Ouvrir l’analyse complète →</Link>
        </div>
        <div className="brand-overview-split">
          <BrandPortfolio items={data.top_brands.slice(0, 5)} compact />
          <aside className="brand-coverage-card">
            <span>Couverture des données marque</span>
            <strong>{formatPercent(data.known_brand_ca_share)}</strong>
            <div><i style={{ width: `${Math.min(data.known_brand_ca_share * 100, 100)}%` }} /></div>
            <p>Part du CA positif reliée à une marque connue. Les références non renseignées restent dans le CA global, mais pas dans le classement.</p>
          </aside>
        </div>
      </section>

      <SectionTitle
        index="04"
        title="Les priorités qui méritent une explication"
        description="Chaque ligne montre le montant concret, le statut et le premier article à discuter."
      />
      <div className="priority-list">
        {data.top_clients.map(client => (
          <article className="priority-row" key={`${client.client_id}-${client.rank}`}>
            <div className="priority-rank"><RankBadge rank={client.rank} /></div>
            <div className="priority-person">
              <strong>{client.client_name}</strong>
              <span className="mono">{client.client_id}</span>
            </div>
            <StatusPill status={client.status} />
            <div className="priority-reason">
              <span>Pourquoi ce rang</span>
              <p>{client.summary}</p>
            </div>
            <div className="priority-article">
              <span>Premier sujet</span>
              <p>{client.top_article ?? 'Ouvrir la conversation sur ses besoins actuels'}</p>
            </div>
            <div className="priority-money">
              <strong>{formatMoney(client.key_amount_tnd, true)}</strong>
              <span>{client.amount_explanation}</span>
            </div>
          </article>
        ))}
      </div>
      <div className="section-action">
        <Link className="primary-button" to="/vendeurs">Explorer chaque portefeuille vendeur</Link>
      </div>

      <div className="dashboard-grid vendor-overview-grid">
        <section className="panel chart-panel">
          <SectionTitle
            index="05"
            title="Portefeuilles les plus exposés"
            description="Montant de priorité cumulé par vendeur."
          />
          <div className="chart-wrap vendor-chart">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={vendorChart} layout="vertical" margin={{ top: 4, right: 14, left: 12, bottom: 4 }}>
                <CartesianGrid stroke="#ebe5db" horizontal={false} />
                <XAxis type="number" hide />
                <YAxis
                  type="category"
                  dataKey="name"
                  width={120}
                  tickLine={false}
                  axisLine={false}
                  tick={{ fontSize: 10 }}
                />
                <Tooltip
                  formatter={(value: number) => [formatMoney(value), 'Montant de priorité']}
                  contentStyle={chartTooltipStyle}
                />
                <Bar dataKey="stake" fill="#e4572e" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="panel status-panel">
          <SectionTitle index="06" title="État de la file clients" />
          <div className="status-stack">
            {data.status_breakdown.map(item => (
              <div key={item.status}>
                <StatusPill status={item.status} />
                <strong>{formatNumber(item.count)}</strong>
                <span>{formatMoney(item.tnd_at_stake, true)} en priorité cumulée</span>
              </div>
            ))}
          </div>
        </section>
      </div>

      <NoticePanel title="Ce que la direction doit lire avant les montants" items={data.notices} />
    </>
  )
}
