import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'

import {
  endpoints,
  formatMoney,
  formatNumber,
  formatPercent,
  getJSON,
  postJSON,
  type ClientDashboard,
  type FeedbackAction,
  type FeedbackEvent,
  type Vendor,
  type VendorDashboard,
} from '../api'
import {
  EmptyState,
  ErrorState,
  Loading,
  PageHeader,
  RankBadge,
  SectionTitle,
  ShowMore,
  StatusPill,
  WhyList,
} from '../components/ui'
import { useFetch } from '../useFetch'

type SessionFeedback = { action?: FeedbackAction; note: string }

const actions: Array<{
  value: FeedbackAction
  label: string
  detail: string
  marker: string
}> = [
  { value: 'CONTACTED', label: 'Client contacté', detail: 'L’échange a eu lieu', marker: '01' },
  { value: 'ORDER', label: 'Commande passée', detail: 'Vente conclue', marker: '02' },
  { value: 'QUOTE', label: 'Préparer un devis', detail: 'Opportunité confirmée', marker: '03' },
  { value: 'SNOOZE', label: 'Reporter', detail: 'À rappeler plus tard', marker: '04' },
  { value: 'UNREACHABLE', label: 'Injoignable', detail: 'Aucun contact établi', marker: '05' },
  { value: 'CLOSED', label: 'Client fermé', detail: 'L’activité a cessé', marker: '06' },
  { value: 'DISMISS', label: 'Non pertinent', detail: 'Recommandation à revoir', marker: '07' },
]

const actionLabels: Record<FeedbackAction, string> = {
  CONTACTED: 'Contacté',
  ORDER: 'Commande',
  QUOTE: 'Devis',
  SNOOZE: 'Reporté',
  UNREACHABLE: 'Injoignable',
  CLOSED: 'Fermé',
  DISMISS: 'Écarté',
}

const OFFER_PAGE = 3

export default function SellerExperiencePage() {
  const vendors = useFetch<{ items: Vendor[] }>(endpoints.vendors)
  const [searchParams, setSearchParams] = useSearchParams()
  const [queueSearch, setQueueSearch] = useState('')
  const [feedback, setFeedback] = useState<Record<string, SessionFeedback>>({})
  const [feedbackError, setFeedbackError] = useState<string | null>(null)

  const requestedVendor = searchParams.get('vendor')
  const requestedClient = searchParams.get('client')
  const activeVendor = vendors.data?.items.some(item => item.vendor_code === requestedVendor)
    ? requestedVendor
    : vendors.data?.items[0]?.vendor_code ?? null
  const dashboard = useFetch<VendorDashboard>(activeVendor ? endpoints.vendorDashboard(activeVendor) : null)
  const activeClient = dashboard.data?.clients.some(item => item.client_id === requestedClient)
    ? requestedClient
    : dashboard.data?.clients[0]?.client_id ?? null
  const client = useFetch<ClientDashboard>(
    activeVendor && activeClient ? endpoints.client(activeVendor, activeClient) : null,
  )
  const [shownOffers, setShownOffers] = useState(OFFER_PAGE)

  useEffect(() => { setShownOffers(OFFER_PAGE) }, [activeVendor, activeClient])

  const visibleQueue = useMemo(() => {
    const normalized = queueSearch.trim().toLocaleLowerCase('fr')
    const items = dashboard.data?.clients ?? []
    return normalized
      ? items.filter(item => `${item.client_name} ${item.client_id}`.toLocaleLowerCase('fr').includes(normalized))
      : items
  }, [dashboard.data, queueSearch])

  const currentIndex = dashboard.data?.clients.findIndex(item => item.client_id === activeClient) ?? -1
  const feedbackKey = activeVendor && activeClient ? `${activeVendor}:${activeClient}` : ''
  const currentFeedback: SessionFeedback = feedback[feedbackKey] ?? { note: '' }
  const processedCount = activeVendor
    ? Object.entries(feedback).filter(([key, value]) => key.startsWith(`${activeVendor}:`) && value.action).length
    : 0
  const queueSize = dashboard.data?.clients.length ?? 0
  const progress = queueSize > 0 ? processedCount / queueSize : 0

  const selectClient = (clientId: string) => {
    if (activeVendor) setSearchParams({ vendor: activeVendor, client: clientId })
  }

  const moveClient = (direction: number) => {
    if (!dashboard.data || !activeVendor || currentIndex < 0) return
    const nextIndex = Math.min(Math.max(currentIndex + direction, 0), dashboard.data.clients.length - 1)
    selectClient(dashboard.data.clients[nextIndex].client_id)
  }

  // Replay: a vendor's progress through a 200-client queue must survive a refresh.
  // Latest event per client wins -- the log keeps every event, the UI shows the last.
  useEffect(() => {
    if (!activeVendor) return
    let cancelled = false
    getJSON<{ items: FeedbackEvent[] }>(endpoints.vendorFeedback(activeVendor))
      .then(data => {
        if (cancelled) return
        const latest: Record<string, SessionFeedback> = {}
        for (const event of data.items) {
          latest[`${event.vendor_code}:${event.client_id}`] =
            { action: event.action, note: event.note }
        }
        setFeedback(latest)
      })
      .catch(() => { /* an empty queue state is better than blocking the page */ })
    return () => { cancelled = true }
  }, [activeVendor])

  const record = (action: FeedbackAction, note: string) => {
    if (!activeVendor || !activeClient) return
    setFeedbackError(null)
    postJSON<FeedbackEvent>(endpoints.feedback, {
      event_id: crypto.randomUUID(),
      vendor_code: activeVendor,
      client_id: activeClient,
      action,
      note,
    }).catch((error: Error) => setFeedbackError(error.message))
  }

  const setAction = (action: FeedbackAction) => {
    if (!feedbackKey) return
    const note = feedback[feedbackKey]?.note ?? ''
    setFeedback(previous => ({ ...previous, [feedbackKey]: { action, note } }))
    record(action, note)
  }

  const setNote = (note: string) => {
    if (!feedbackKey) return
    setFeedback(previous => ({
      ...previous,
      [feedbackKey]: { action: previous[feedbackKey]?.action, note },
    }))
  }

  if (vendors.error) return <ErrorState error={vendors.error} />
  if (vendors.loading || !vendors.data) return <Loading label="Préparation de l’expérience vendeur" />

  return (
    <>
      {feedbackError ? (
        <aside className="notice-panel" role="alert">
          <div className="notice-icon">!</div>
          <div><strong>Enregistrement impossible</strong>
            <p>{feedbackError}</p></div>
        </aside>
      ) : null}
      <PageHeader
        eyebrow="Mode terrain · simulation"
        title="Tester la recommandation comme un vendeur."
        description="Parcourez la file, vérifiez les preuves, préparez l’offre et simulez la décision que le vendeur prendrait réellement face au client."
        action={
          <label className="select-control">
            <span>Vendeur en simulation</span>
            <select
              value={activeVendor ?? ''}
              onChange={event => {
                setSearchParams({ vendor: event.target.value })
                setQueueSearch('')
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
      {dashboard.loading || !dashboard.data ? <Loading label="Chargement de la file vendeur" /> : (
        <>
          <section className="seller-session-strip">
            <div className="seller-session-copy">
              <span>Session de test locale</span>
              <strong>{dashboard.data.vendor_name}</strong>
              <p>Les décisions ci-dessous sont enregistrées dans le journal des retours dès qu’une action est choisie.</p>
            </div>
            <div className="seller-progress">
              <div><span>Progression</span><strong>{processedCount} / {queueSize}</strong></div>
              <div className="seller-progress-track"><i style={{ width: `${progress * 100}%` }} /></div>
            </div>
            <div className="seller-session-stat"><span>Montant de priorité</span><strong>{formatMoney(dashboard.data.tnd_at_stake, true)}</strong></div>
            <div className="seller-session-stat"><span>CA marque couvert</span><strong>{formatPercent(dashboard.data.known_brand_ca_share)}</strong></div>
          </section>

          <div className="seller-experience-grid">
            <aside className="seller-queue-panel">
              <div className="seller-queue-head">
                <div><span>File de travail</span><strong>{formatNumber(visibleQueue.length)} clients</strong></div>
                <input
                  type="search"
                  value={queueSearch}
                  placeholder="Chercher un client"
                  onChange={event => setQueueSearch(event.target.value)}
                  aria-label="Chercher dans la file vendeur"
                />
              </div>
              <div className="seller-queue-list">
                {visibleQueue.map(item => {
                  const itemFeedback = feedback[`${activeVendor}:${item.client_id}`]
                  return (
                    <button
                      key={item.client_id}
                      className={`${item.client_id === activeClient ? 'selected' : ''} ${itemFeedback?.action ? 'processed' : ''}`}
                      onClick={() => selectClient(item.client_id)}
                    >
                      <RankBadge rank={item.rank} />
                      <div>
                        <strong>{item.client_name}</strong>
                        <span className="mono">{item.client_id}</span>
                        <small>{item.summary}</small>
                      </div>
                      <div className="seller-queue-value">
                        <strong>{formatMoney(item.key_amount_tnd, true)}</strong>
                        {itemFeedback?.action
                          ? <span className={`feedback-label ${itemFeedback.action.toLowerCase()}`}>{actionLabels[itemFeedback.action]}</span>
                          : <StatusPill status={item.status} />}
                      </div>
                    </button>
                  )
                })}
              </div>
            </aside>

            <section className="seller-focus-zone">
              {client.error ? <ErrorState error={client.error} /> : null}
              {client.loading || !client.data ? <Loading label="Préparation de la recommandation" /> : (
                <>
                  <article className="seller-client-card">
                    <header className="seller-client-head">
                      <div>
                        <span className="seller-step">Client {currentIndex + 1} sur {queueSize}</span>
                        <h2>{client.data.client_name}</h2>
                        <p><span className="mono">{client.data.client_id}</span> · dernière activité {client.data.last_activity ?? 'non renseignée'}</p>
                      </div>
                      <StatusPill status={client.data.status} />
                    </header>

                    <section className="seller-brief">
                      <div>
                        <span>Montant à expliquer</span>
                        <strong>{formatMoney(client.data.key_amount_tnd)}</strong>
                        <small>{client.data.amount_explanation}</small>
                      </div>
                      <p>{client.data.summary}</p>
                    </section>

                    <div className="seller-evidence-grid">
                      <section className="seller-evidence-block">
                        <SectionTitle title="Pourquoi ce rang" />
                        <WhyList items={client.data.why_recommended} />
                      </section>
                      <section className="seller-profile-block">
                        <SectionTitle title="Repères avant l’appel" />
                        <div className="seller-profile-stats">
                          <div><span>Valeur attendue</span><strong>{formatMoney(client.data.expected_value_tnd, true)}</strong></div>
                          <div><span>Baisse constatée</span><strong>{formatMoney(client.data.lost_revenue_tnd, true)}</strong></div>
                          <div><span>CA marque connu</span><strong>{formatPercent(client.data.known_brand_ca_share)}</strong></div>
                          <div><span>Top 3 marques</span><strong>{formatPercent(client.data.brand_concentration_top3)}</strong></div>
                        </div>
                        <div className="seller-brand-chips">
                          {client.data.brands.slice(0, 4).map(brand => (
                            <span key={brand.brand}>{brand.brand} · {formatMoney(brand.ca_tnd, true)}</span>
                          ))}
                        </div>
                      </section>
                    </div>

                    <SectionTitle
                      title="Offres à présenter au client"
                      description="Les trois meilleures d’abord, le reste de son catalogue à la demande."
                    />
                    {client.data.opportunities.length === 0 ? (
                      <EmptyState title="Aucun article précis">Commencez par qualifier le besoin actuel du client.</EmptyState>
                    ) : (
                      <div className="seller-offer-grid">
                        {client.data.opportunities.slice(0, shownOffers).map(item => (
                          <article key={item.article_id}>
                            <div className="seller-offer-head">
                              <RankBadge rank={item.rank} />
                              <span className="mono">{item.article_id}</span>
                              {item.brand ? <span className="brand-tag">{item.brand}</span> : null}
                            </div>
                            <h3>{item.designation}</h3>
                            <div className="seller-offer-values">
                              <div><span>CA précédent</span><strong>{formatMoney(item.previous_ca_tnd, true)}</strong></div>
                              <div><span>Valeur attendue</span><strong>{formatMoney(item.expected_value_tnd, true)}</strong></div>
                              <div><span>Score brut</span><strong>{formatPercent(item.repurchase_score_raw, 0)}</strong></div>
                            </div>
                            <div className="seller-offer-reason"><span>Argument vendeur</span><p>{item.reason}</p></div>
                          </article>
                        ))}
                        <ShowMore
                          shown={Math.min(shownOffers, client.data.opportunities.length)}
                          total={client.data.opportunities.length}
                          initial={OFFER_PAGE}
                          step={OFFER_PAGE}
                          onChange={setShownOffers}
                        />
                      </div>
                    )}

                    {client.data.cross_sell_suggestion ? (
                      <div className="seller-cross-sell">
                        <span>Suggestion complémentaire</span>
                        <strong>{client.data.cross_sell_suggestion}</strong>
                      </div>
                    ) : null}
                  </article>

                  <section className="seller-decision-panel">
                    <div className="seller-decision-head">
                      <div><span>Retour terrain</span><h2>Que ferait le vendeur ?</h2></div>
                      {currentFeedback.action ? <strong>Décision : {actionLabels[currentFeedback.action]}</strong> : <small>Aucune décision testée</small>}
                    </div>
                    <div className="seller-action-grid">
                      {actions.map(action => (
                        <button
                          key={action.value}
                          className={`${currentFeedback.action === action.value ? 'active' : ''} action-${action.value.toLowerCase()}`}
                          onClick={() => setAction(action.value)}
                        >
                          <span>{action.marker}</span>
                          <strong>{action.label}</strong>
                          <small>{action.detail}</small>
                        </button>
                      ))}
                    </div>
                    <label className="seller-note-field">
                      <span>Note du vendeur pour tester l’expérience</span>
                      <textarea
                        value={currentFeedback.note}
                        placeholder="Ex. Le client veut être rappelé mardi, vérifier le stock avant le devis…"
                        onChange={event => setNote(event.target.value)}
                        onBlur={() => {
                          if (currentFeedback.action) record(currentFeedback.action, currentFeedback.note)
                        }}
                      />
                    </label>
                    <div className="seller-navigation">
                      <button disabled={currentIndex <= 0} onClick={() => moveClient(-1)}>← Client précédent</button>
                      <span>Les retours sont enregistrés dès qu’une action est choisie.</span>
                      <button disabled={currentIndex >= queueSize - 1} onClick={() => moveClient(1)}>Client suivant →</button>
                    </div>
                  </section>
                </>
              )}
            </section>
          </div>
        </>
      )}
    </>
  )
}
