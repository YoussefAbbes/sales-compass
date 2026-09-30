import { useMemo, useState } from 'react'
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
  type BrandDashboard,
  type BrandList,
  type BrandRankItem,
} from '../api'
import {
  chartTooltipStyle,
  EmptyState,
  ErrorState,
  Loading,
  MetricCard,
  NoticePanel,
  PageHeader,
  RankBadge,
  SectionTitle,
  TrendPill,
} from '../components/ui'
import { useFetch } from '../useFetch'

function EntityRanking({ title, description, items, entity }: {
  title: string
  description: string
  items: BrandRankItem[]
  entity: 'vendor' | 'client' | 'article'
}) {
  if (items.length === 0) {
    return <EmptyState title={`Aucun ${entity}`}>Aucune activité positive n’est reliée à cette marque.</EmptyState>
  }
  const maximum = Math.max(...items.map(item => item.ca_tnd), 1)
  return (
    <section className="panel entity-ranking">
      <SectionTitle title={title} description={description} />
      <div className="entity-list">
        {items.map(item => (
          <article key={`${entity}-${item.code}`}>
            <RankBadge rank={item.rank} />
            <div className="entity-main">
              <strong>{item.name}</strong>
              <span className="mono">{item.code}</span>
              <div className="entity-bar"><i style={{ width: `${Math.max((item.ca_tnd / maximum) * 100, 2)}%` }} /></div>
            </div>
            <div className="entity-value">
              <strong>{formatMoney(item.ca_tnd, true)}</strong>
              <span>{formatPercent(item.share_of_brand_ca)} du CA marque</span>
              {item.clients !== null ? <small>{formatNumber(item.clients)} client{item.clients > 1 ? 's' : ''}</small> : null}
              {item.articles !== null ? <small>{formatNumber(item.articles)} réf.</small> : null}
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}

export default function BrandsPage() {
  const brands = useFetch<BrandList>(endpoints.brands())
  const [selectedBrand, setSelectedBrand] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const activeBrand = selectedBrand ?? brands.data?.items[0]?.brand ?? null
  const detail = useFetch<BrandDashboard>(activeBrand ? endpoints.brandDetail(activeBrand) : null)

  const visibleBrands = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase('fr')
    const items = brands.data?.items ?? []
    return normalized ? items.filter(item => item.brand.toLocaleLowerCase('fr').includes(normalized)) : items
  }, [brands.data, query])

  if (brands.error) return <ErrorState error={brands.error} />
  if (brands.loading || !brands.data) return <Loading label="Cartographie des marques" />

  // Read the comparison off the API instead of recomputing it. `yearly_ca_tnd`
  // holds whole-year totals, and the ledger's last year is partial, so dividing
  // one by the other reported a ~50% collapse for every brand. The API restricts
  // both years to the months they share before dividing.
  const activeSummary = brands.data.items.find(item => item.brand === activeBrand) ?? null
  const yearly2025 = activeSummary?.ca_2025_tnd ?? 0
  const yearly2026 = activeSummary?.ca_2026_tnd ?? 0
  const evolution = activeSummary?.evolution_pct ?? null

  return (
    <>
      <PageHeader
        eyebrow="Marque → vendeur → client → article"
        title="Voir qui travaille avec quelle marque."
        description="Une lecture croisée du portefeuille : poids de chaque marque, vendeurs qui la développent, clients qui l’achètent et références qui portent le chiffre d’affaires."
        action={<span className="coverage-badge"><strong>{formatPercent(brands.data.known_brand_ca_share)}</strong><span>du CA relié à une marque connue</span></span>}
      />

      <section className="brand-directory">
        <div className="brand-directory-head">
          <div>
            <span className="eyebrow">Répertoire observé</span>
            <strong>{formatNumber(brands.data.total_brands)} marques actives</strong>
          </div>
          <input
            type="search"
            value={query}
            placeholder="Chercher une marque"
            onChange={event => setQuery(event.target.value)}
            aria-label="Chercher une marque"
          />
        </div>
        <div className="brand-tabs" role="list">
          {visibleBrands.map(item => (
            <button
              key={item.brand}
              className={activeBrand === item.brand ? 'active' : ''}
              onClick={() => setSelectedBrand(item.brand)}
            >
              <span>{String(item.rank).padStart(2, '0')}</span>
              <strong>{item.brand}</strong>
              <small>{formatMoney(item.ca_tnd, true)}</small>
            </button>
          ))}
          {visibleBrands.length === 0 ? <p>Aucune marque ne correspond à cette recherche.</p> : null}
        </div>
      </section>

      {detail.error ? <ErrorState error={detail.error} /> : null}
      {detail.loading || !detail.data ? <Loading label="Analyse de la marque sélectionnée" /> : (
        <>
          <section className="brand-hero">
            <div className="brand-hero-copy">
              <span>Marque sélectionnée</span>
              <h2>{detail.data.brand}</h2>
              <p>
                {formatPercent(detail.data.share_of_known_brand_ca)} du CA attribué aux marques connues,
                réparti sur {formatNumber(detail.data.purchase_months)} mois d’activité.
              </p>
            </div>
            <div className="brand-hero-number">
              <strong>{formatMoney(detail.data.ca_tnd, true)}</strong>
              <span>CA positif observé</span>
            </div>
          </section>

          <div className="metric-grid brand-metrics">
            <MetricCard label="Clients acheteurs" value={formatNumber(detail.data.clients)} detail="clients distincts observés" tone="pine" />
            <MetricCard label="Vendeurs concernés" value={formatNumber(detail.data.vendors)} detail="portefeuilles ayant vendu la marque" tone="clay" />
            <MetricCard label="Références vendues" value={formatNumber(detail.data.articles)} detail="articles distincts avec CA positif" tone="gold" />
            <MetricCard label="Poids de la marque" value={formatPercent(detail.data.share_of_known_brand_ca)} detail="parmi le CA de marques connues" />
          </div>

          <div className="dashboard-grid brand-analysis-grid">
            <section className="panel chart-panel">
              <SectionTitle
                index="01"
                title="Rythme mensuel de la marque"
                description="CA réellement observé, sans compléter les mois absents."
              />
              <div className="chart-wrap brand-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={detail.data.monthly_ca} margin={{ top: 12, right: 16, left: 4, bottom: 4 }}>
                    <defs>
                      <linearGradient id="brandFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#e4572e" stopOpacity={0.3} />
                        <stop offset="100%" stopColor="#e4572e" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#ebe5db" vertical={false} />
                    <XAxis dataKey="month" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} />
                    <YAxis
                      tickFormatter={(value: number) => formatMoney(value, true).replace(' TND', '')}
                      tickLine={false}
                      axisLine={false}
                      width={56}
                      tick={{ fontSize: 10 }}
                    />
                    <Tooltip formatter={(value: number) => [formatMoney(value), 'CA marque']} contentStyle={chartTooltipStyle} />
                    <Area type="monotone" dataKey="ca_tnd" stroke="#e4572e" strokeWidth={3} fill="url(#brandFill)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section className="panel brand-year-panel">
              <SectionTitle index="02" title="Évolution comparable" description="À périmètre de mois identique, année contre année." />
              <div className="year-comparison">
                <div><span>2025</span><strong>{formatMoney(yearly2025, true)}</strong></div>
                <div className="year-arrow">→</div>
                <div><span>2026</span><strong>{formatMoney(yearly2026, true)}</strong></div>
              </div>
              <div className="evolution-callout">
                <TrendPill value={evolution} />
                <p>{evolution === null ? 'Pas de base 2025 : la marque apparaît comme nouvelle sur la période.' : 'Écart du CA sur les mois que les deux années ont en commun.'}</p>
              </div>
              <div className="share-meter">
                <span>Part dans le CA des marques connues</span>
                <div><i style={{ width: `${Math.min(detail.data.share_of_known_brand_ca * 100, 100)}%` }} /></div>
                <strong>{formatPercent(detail.data.share_of_known_brand_ca)}</strong>
              </div>
            </section>
          </div>

          <SectionTitle
            index="03"
            title="Qui fait réellement vivre cette marque ?"
            description="Les trois angles se répondent : portefeuille vendeur, relation client et références achetées."
          />
          <div className="brand-ranking-grid">
            <EntityRanking title="Top vendeurs" description="Portefeuilles avec le plus de CA marque." items={detail.data.top_vendors} entity="vendor" />
            <EntityRanking title="Top clients" description="Clients les plus engagés avec la marque." items={detail.data.top_clients} entity="client" />
            <EntityRanking title="Top articles" description="Références qui portent le CA de la marque." items={detail.data.top_articles} entity="article" />
          </div>

          <NoticePanel title="Comment lire les statistiques par marque" items={detail.data.notices} />
        </>
      )}
    </>
  )
}
