import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
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
  type ClientDashboard,
  type Vendor,
  type VendorDashboard,
} from '../api'
import {
  BrandPortfolio,
  chartTooltipStyle,
  EmptyState,
  ErrorState,
  Loading,
  MoneyBreakdown,
  NoticePanel,
  PageHeader,
  RankBadge,
  SectionTitle,
  ShowMore,
  StatusPill,
  WhyList,
} from '../components/ui'
import { useFetch } from '../useFetch'

const ARTICLE_PAGE = 3

function ClientDossier({ vendor, client }: { vendor: string; client: string }) {
  const { data, error, loading } = useFetch<ClientDashboard>(endpoints.client(vendor, client))
  // The dossier now carries every scored article for this client; three is the
  // opening view, not the ceiling it used to be.
  const [shown, setShown] = useState(ARTICLE_PAGE)

  useEffect(() => { setShown(ARTICLE_PAGE) }, [vendor, client])

  if (error) return <ErrorState error={error} />
  if (loading || !data) return <Loading label="Construction du dossier client" />

  return (
    <article className="client-dossier">
      <div className="dossier-head">
        <div>
          <div className="dossier-kicker"><RankBadge rank={data.rank} /> dans la file du vendeur</div>
          <h2>{data.client_name}</h2>
          <p><span className="mono">{data.client_id}</span> · {data.vendor_name}</p>
        </div>
        <StatusPill status={data.status} />
      </div>

      <div className="key-explanation">
        <span>Le chiffre concret à retenir</span>
        <strong>{formatMoney(data.key_amount_tnd)}</strong>
        <p>{data.amount_explanation}. {data.summary}</p>
      </div>

      <MoneyBreakdown expected={data.expected_value_tnd} lost={data.lost_revenue_tnd} />

      <SectionTitle
        title="Marques avec lesquelles ce client travaille"
        description="Affinité historique chez ce vendeur : poids du CA, références achetées et évolution annuelle."
      />
      <div className="client-brand-summary">
        <div><span>CA avec marque connue</span><strong>{formatPercent(data.known_brand_ca_share)}</strong></div>
        <div><span>Concentration top 3</span><strong>{formatPercent(data.brand_concentration_top3)}</strong></div>
      </div>
      <BrandPortfolio items={data.brands.slice(0, 6)} compact />

      <SectionTitle title="Pourquoi ce client est recommandé" />
      <WhyList items={data.why_recommended} />

      <SectionTitle
        title="Articles à proposer"
        description="Classés par valeur attendue = score brut × CA précédent."
      />
      {data.opportunities.length === 0 ? (
        <EmptyState title="Aucun article précis à proposer">
          Ouvrez la conversation sur ses besoins actuels et son activité.
        </EmptyState>
      ) : (
        <div className="opportunity-grid">
          {data.opportunities.slice(0, shown).map(opportunity => (
            <article className="opportunity-card" key={opportunity.article_id}>
              <div className="opportunity-top">
                <RankBadge rank={opportunity.rank} />
                <span className="mono">{opportunity.article_id}</span>
                {opportunity.brand ? <span className="brand-tag">{opportunity.brand}</span> : null}
              </div>
              <h3>{opportunity.designation}</h3>
              <div className="opportunity-stats">
                <div><span>Score brut</span><strong>{formatPercent(opportunity.repurchase_score_raw, 0)}</strong></div>
                <div><span>CA précédent</span><strong>{formatMoney(opportunity.previous_ca_tnd, true)}</strong></div>
                <div><span>Valeur attendue</span><strong>{formatMoney(opportunity.expected_value_tnd, true)}</strong></div>
              </div>
              <div className="why-box"><span>Pourquoi cette pièce</span><p>{opportunity.reason}</p></div>
            </article>
          ))}
          <ShowMore
            shown={Math.min(shown, data.opportunities.length)}
            total={data.opportunities.length}
            initial={ARTICLE_PAGE}
            step={ARTICLE_PAGE}
            onChange={setShown}
          />
        </div>
      )}

      {data.cross_sell_suggestion ? (
        <div className="cross-sell">
          <span>À glisser si l’échange se passe bien</span>
          <strong>{data.cross_sell_suggestion}</strong>
          <p>Suggestion de vente complémentaire issue des paniers observés.</p>
        </div>
      ) : null}

      <SectionTitle title="Historique chez ce vendeur" description="CA mensuel sur toute la profondeur du registre." />
      <div className="year-chips">
        {Object.entries(data.yearly_ca_tnd).map(([year, value]) => (
          <div key={year}><span>{year}</span><strong>{formatMoney(value, true)}</strong></div>
        ))}
      </div>
      <div className="chart-wrap dossier-chart">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data.monthly_ca} margin={{ top: 12, right: 16, left: 4, bottom: 4 }}>
            <defs>
              <linearGradient id="clientFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#0f5c4a" stopOpacity={0.28} />
                <stop offset="100%" stopColor="#0f5c4a" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="#ebe5db" vertical={false} />
            <XAxis dataKey="month" tickLine={false} axisLine={false} tick={{ fontSize: 10 }} />
            <YAxis
              tickFormatter={(value: number) => formatMoney(value, true).replace(' TND', '')}
              tickLine={false}
              axisLine={false}
              width={54}
              tick={{ fontSize: 10 }}
            />
            <Tooltip formatter={(value: number) => [formatMoney(value), 'CA']} contentStyle={chartTooltipStyle} />
            <Area type="monotone" dataKey="ca_tnd" stroke="#0f5c4a" strokeWidth={2.5} fill="url(#clientFill)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <NoticePanel title="Interprétation de cette fiche" items={data.notices} />
    </article>
  )
}

export default function ClientsPage() {
  const vendors = useFetch<{ items: Vendor[] }>(endpoints.vendors)
  const [searchParams, setSearchParams] = useSearchParams()
  const [search, setSearch] = useState('')
  const [status, setStatus] = useState('ALL')

  const requestedVendor = searchParams.get('vendor')
  const requestedClient = searchParams.get('client')
  const activeVendor = vendors.data?.items.some(item => item.vendor_code === requestedVendor)
    ? requestedVendor
    : vendors.data?.items[0]?.vendor_code ?? null
  const dashboard = useFetch<VendorDashboard>(activeVendor ? endpoints.vendorDashboard(activeVendor) : null)
  const activeClient = dashboard.data?.clients.some(item => item.client_id === requestedClient)
    ? requestedClient
    : dashboard.data?.clients[0]?.client_id ?? null

  const visibleClients = useMemo(() => {
    const clients = dashboard.data?.clients ?? []
    const query = search.trim().toLowerCase()
    return clients.filter(client => {
      const matchesSearch = !query
        || client.client_name.toLowerCase().includes(query)
        || client.client_id.toLowerCase().includes(query)
      return matchesSearch && (status === 'ALL' || client.status === status)
    })
  }, [dashboard.data, search, status])

  if (vendors.error) return <ErrorState error={vendors.error} />
  if (vendors.loading || !vendors.data) return <Loading label="Chargement des clients" />

  return (
    <>
      <PageHeader
        eyebrow="Dossiers clients"
        title="Un client, ses marques, ses recommandations."
        description="Cherchez un client dans le portefeuille sélectionné puis ouvrez son dossier : historique, montants, marques achetées, raisons et articles à proposer."
        action={
          <label className="select-control">
            <span>Portefeuille vendeur</span>
            <select
              value={activeVendor ?? ''}
              onChange={event => {
                setSearchParams({ vendor: event.target.value })
                setSearch('')
                setStatus('ALL')
              }}
            >
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
      {dashboard.loading || !dashboard.data ? <Loading label="Préparation des dossiers clients" /> : (
        <>
          <div className="client-context-strip">
            <div><span>Portefeuille</span><strong>{dashboard.data.vendor_name}</strong></div>
            <div><span>Clients prioritaires</span><strong>{formatNumber(dashboard.data.queued_clients)}</strong></div>
            <div><span>CA avec marque connue</span><strong>{formatPercent(dashboard.data.known_brand_ca_share)}</strong></div>
            <div><span>Montant de priorité</span><strong>{formatMoney(dashboard.data.tnd_at_stake, true)}</strong></div>
          </div>

          <div className="vendor-workbench client-only-workbench">
            <aside className="client-browser">
              <div className="browser-head">
                <div><strong>Clients</strong><span>{visibleClients.length} affiché(s)</span></div>
                <input
                  type="search"
                  value={search}
                  placeholder="Nom ou code client"
                  onChange={event => setSearch(event.target.value)}
                  aria-label="Chercher un client"
                />
              </div>
              <div className="filter-tabs">
                {[
                  ['ALL', 'Tous'],
                  ['INACTIVE_2026', 'Inactifs'],
                  ['DECLINING', 'En baisse'],
                  ['ACTIVE', 'Actifs'],
                ].map(([value, label]) => (
                  <button key={value} className={status === value ? 'active' : ''} onClick={() => setStatus(value)}>
                    {label}
                  </button>
                ))}
              </div>
              <div className="client-list">
                {visibleClients.map(client => (
                  <button
                    key={client.client_id}
                    className={activeClient === client.client_id ? 'selected' : ''}
                    onClick={() => setSearchParams({ vendor: activeVendor ?? '', client: client.client_id })}
                  >
                    <div className="client-list-main">
                      <RankBadge rank={client.rank} />
                      <div><strong>{client.client_name}</strong><span className="mono">{client.client_id}</span></div>
                    </div>
                    <StatusPill status={client.status} />
                    <div className="client-list-bottom">
                      <span>{client.amount_explanation}</span>
                      <strong>{formatMoney(client.key_amount_tnd, true)}</strong>
                    </div>
                  </button>
                ))}
                {visibleClients.length === 0 ? (
                  <EmptyState title="Aucun client ne correspond">Modifiez la recherche ou le statut.</EmptyState>
                ) : null}
              </div>
            </aside>

            <section className="dossier-zone">
              {activeVendor && activeClient ? (
                <ClientDossier vendor={activeVendor} client={activeClient} />
              ) : (
                <EmptyState title="Choisissez un client">Sa fiche expliquée apparaîtra ici.</EmptyState>
              )}
            </section>
          </div>
        </>
      )}
    </>
  )
}
