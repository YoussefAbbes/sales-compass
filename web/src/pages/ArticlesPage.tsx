import { useEffect, useState } from 'react'

import {
  endpoints,
  formatMoney,
  formatPercent,
  formatRatio,
  getJSON,
  type Article,
  type ArticleRecommendations,
  type ArticleSearch,
  type Vendor,
} from '../api'
import {
  EmptyState,
  ErrorState,
  Formula,
  Loading,
  MetricCard,
  PageHeader,
  RankBadge,
  SectionTitle,
  ShowMore,
} from '../components/ui'
import { useFetch } from '../useFetch'

function emptyReason(reason: string | null) {
  switch (reason) {
    case 'insufficient_current_buyers': return 'Pas assez d’acheteurs récents pour construire un profil fiable.'
    case 'no_signature_articles': return 'Aucun article-repère suffisamment spécifique n’a été trouvé.'
    case 'no_eligible_clients': return 'Tous les clients éligibles achètent déjà cette pièce.'
    case 'no_basket_overlap': return 'Des prospects existent, mais leurs paniers ne fournissent pas assez de preuves.'
    default: return 'Aucune recommandation disponible pour cette sélection.'
  }
}

const PAGE = 8

export default function ArticlesPage() {
  const vendors = useFetch<{ items: Vendor[] }>(endpoints.vendors)
  const [vendorCode, setVendorCode] = useState<string | null>(null)
  const [query, setQuery] = useState('01381')
  const [results, setResults] = useState<Article[]>([])
  const [searching, setSearching] = useState(true)
  const [searchError, setSearchError] = useState<unknown>(null)
  const [articleId, setArticleId] = useState<string | null>(null)
  // Both blocks arrive fully ranked in one response; revealing is local so a
  // click costs no round-trip and the ranks stay absolute.
  const [shownBuyers, setShownBuyers] = useState(PAGE)
  const [shownProspects, setShownProspects] = useState(PAGE)

  useEffect(() => {
    setShownBuyers(PAGE)
    setShownProspects(PAGE)
  }, [articleId, vendorCode])

  useEffect(() => {
    const controller = new AbortController()
    const timer = window.setTimeout(() => {
      setSearching(true)
      getJSON<ArticleSearch>(endpoints.articles(query), controller.signal)
        .then(response => {
          setResults(response.items)
          setSearchError(null)
          setSearching(false)
          setArticleId(current => {
            if (current && response.items.some(item => item.article_id === current)) return current
            return response.items[0]?.article_id ?? null
          })
        })
        .catch(error => {
          if (!controller.signal.aborted) {
            setSearchError(error)
            setSearching(false)
          }
        })
    }, 220)
    return () => {
      window.clearTimeout(timer)
      controller.abort()
    }
  }, [query])

  const activeVendor = vendorCode ?? vendors.data?.items[0]?.vendor_code ?? null
  const recommendations = useFetch<ArticleRecommendations>(
    activeVendor && articleId ? endpoints.recommendations(activeVendor, articleId) : null,
  )

  if (vendors.error) return <ErrorState error={vendors.error} />
  if (vendors.loading || !vendors.data) return <Loading label="Chargement du catalogue" />

  return (
    <>
      <PageHeader
        eyebrow="Article → clients"
        title="Pour cette pièce, qui faut-il appeler ?"
        description="Le tableau sépare volontairement les anciens acheteurs des nouveaux prospects : ce sont deux populations et deux preuves différentes."
      />

      <section className="article-search-panel">
        <label>
          <span>1. Vendeur</span>
          <select value={activeVendor ?? ''} onChange={event => setVendorCode(event.target.value)}>
            {vendors.data.items.map(vendor => (
              <option key={vendor.vendor_code} value={vendor.vendor_code}>{vendor.vendor_name}</option>
            ))}
          </select>
        </label>
        <label className="article-query">
          <span>2. Rechercher une pièce</span>
          <input
            type="search"
            value={query}
            placeholder="Référence, marque ou désignation…"
            onChange={event => setQuery(event.target.value)}
          />
        </label>
        <label>
          <span>3. Article</span>
          <select
            value={articleId ?? ''}
            onChange={event => setArticleId(event.target.value)}
            disabled={searching || results.length === 0}
          >
            {results.map(article => (
              <option key={article.article_id} value={article.article_id}>{article.label}</option>
            ))}
          </select>
        </label>
      </section>

      {searchError ? <ErrorState error={searchError} /> : null}
      {!searching && results.length === 0 ? (
        <EmptyState title="Aucune pièce trouvée">Essayez une référence plus courte, une marque ou un autre mot.</EmptyState>
      ) : null}
      {recommendations.error ? <ErrorState error={recommendations.error} /> : null}
      {recommendations.loading ? <Loading label="Calcul des acheteurs et prospects" /> : null}

      {recommendations.data ? (
        <>
          <div className="selected-article">
            <div>
              <span className="eyebrow">Article sélectionné</span>
              <h2>{recommendations.data.article.designation}</h2>
              <p>
                <span className="mono">{recommendations.data.article.article_id}</span>
                {recommendations.data.article.brand ? ` · ${recommendations.data.article.brand}` : ''}
              </p>
            </div>
            <div className="article-coverage">
              <span>{recommendations.data.prospects.current_window_buyers} acheteurs récents</span>
              <span>{recommendations.data.prospects.eligible_candidates} prospects éligibles</span>
              <span>source : {recommendations.data.prospects.source === 'model' ? 'modèle CatBoost' : 'heuristique panier'}</span>
            </div>
          </div>

          <div className="two-model-columns">
            <section className="recommendation-column">
              <SectionTitle
                index="A"
                title="Ils l’achetaient"
                description="Clients avec historique sur cette pièce, classés par argent en jeu."
              />
              <Formula>
                <span>Valeur attendue</span>
                <strong>=</strong>
                <span>score brut de rachat</span>
                <strong>×</strong>
                <span>CA précédent</span>
              </Formula>
              {recommendations.data.repurchases.items.length === 0 ? (
                <EmptyState title="Aucun ancien acheteur dans ce portefeuille">
                  Passez au bloc prospects pour chercher de nouveaux clients.
                </EmptyState>
              ) : (
                <div className="article-results">
                  {recommendations.data.repurchases.items.slice(0, shownBuyers).map(item => (
                    <article key={item.client_id} className="result-card repurchase-card">
                      <div className="result-head">
                        <RankBadge rank={item.rank} />
                        <div><strong>{item.client_name}</strong><span className="mono">{item.client_id}</span></div>
                        <strong className="result-value">{formatMoney(item.expected_value_tnd, true)}</strong>
                      </div>
                      <div className="result-stats">
                        <MetricCard
                          label="Score brut"
                          value={formatPercent(item.repurchase_score_raw, 0)}
                          tone="pine"
                        />
                        <MetricCard
                          label="CA précédent"
                          value={formatMoney(item.previous_ca_tnd, true)}
                          detail="montant historique observé"
                        />
                      </div>
                      <div className="signal-grid">
                        <div><strong>{item.signals.n_orders}</strong><span>achats de la pièce</span></div>
                        <div><strong>{item.signals.days_since_last}</strong><span>jours depuis le dernier</span></div>
                        <div>
                          <strong>{item.signals.median_interval_days === null ? '—' : Math.round(item.signals.median_interval_days)}</strong>
                          <span>rythme habituel (j)</span>
                        </div>
                        <div>
                          <strong>{formatRatio(item.signals.overdue_ratio)}</strong>
                          <span>de son rythme</span>
                        </div>
                      </div>
                      <div className="why-box">
                        <span>Pourquoi ce rang</span>
                        <p>{item.reason}</p>
                      </div>
                    </article>
                  ))}
                  <ShowMore
                    shown={Math.min(shownBuyers, recommendations.data.repurchases.items.length)}
                    total={recommendations.data.repurchases.items.length}
                    initial={PAGE}
                    step={PAGE}
                    onChange={setShownBuyers}
                  />
                </div>
              )}
            </section>

            <section className="recommendation-column prospect-column">
              <SectionTitle
                index="B"
                title="À prospecter"
                description="Clients qui n’ont jamais acheté cette pièce, classés par ressemblance factuelle."
              />
              <Formula>
                <span>Panier du client</span>
                <strong>↔</strong>
                <span>profil des acheteurs</span>
                <strong>+</strong>
                <span>marque & famille</span>
              </Formula>
              {recommendations.data.prospects.items.length === 0 ? (
                <EmptyState title="Aucun prospect classable">
                  {emptyReason(recommendations.data.prospects.empty_reason)}
                </EmptyState>
              ) : (
                <div className="article-results">
                  {recommendations.data.prospects.items.slice(0, shownProspects).map(item => (
                    <article key={item.client_id} className="result-card prospect-card">
                      <div className="result-head">
                        <RankBadge rank={item.rank} />
                        <div><strong>{item.client_name}</strong><span className="mono">{item.client_id}</span></div>
                        <strong className="result-value">{formatMoney(item.client_ca_tnd, true)} CA</strong>
                      </div>
                      <div className="signal-grid">
                        <div><strong>{item.signals.similar_buyers_out_of_20}</strong><span>voisins acheteurs / 20</span></div>
                        <div><strong>{item.signals.signature_articles_owned}/{item.signals.signature_articles_total}</strong><span>articles-repères détenus</span></div>
                        <div><strong>{item.signals.same_brand_articles_owned}</strong><span>articles même marque</span></div>
                        <div><strong>{item.signals.same_family_articles_owned}</strong><span>articles même famille</span></div>
                      </div>
                      <div className="why-box"><span>Pourquoi ce prospect</span><p>{item.reason}</p></div>
                    </article>
                  ))}
                  <ShowMore
                    shown={Math.min(shownProspects, recommendations.data.prospects.items.length)}
                    total={recommendations.data.prospects.items.length}
                    initial={PAGE}
                    step={PAGE}
                    onChange={setShownProspects}
                  />
                </div>
              )}
            </section>
          </div>
        </>
      ) : null}
    </>
  )
}
