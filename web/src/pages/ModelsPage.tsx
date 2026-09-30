import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { endpoints, formatPercent, type Overview } from '../api'
import {
  chartTooltipStyle,
  ErrorState,
  Formula,
  Loading,
  NoticePanel,
  PageHeader,
  SectionTitle,
} from '../components/ui'
import { useFetch } from '../useFetch'

// Both models name their own features, and the two vocabularies barely overlap.
// A key missing here falls back to the raw snake_case name, so this map has to
// track the models rather than the other way round.
const labels: Record<string, string> = {
  // --- rolling repurchase model ---
  amount_sum: 'CA cumulé du couple',
  qty_sum: 'Quantité cumulée',
  n_orders: 'Nombre de commandes',
  n_months: 'Mois d’achat du couple',
  days_since_last: 'Jours depuis le dernier achat',
  days_since_first: 'Jours depuis le premier achat',
  median_interval: 'Intervalle médian entre achats',
  interval_cv: 'Régularité des intervalles',
  amount_per_order: 'Montant moyen par commande',
  return_rate: 'Taux de retour',
  overdue_ratio: 'Retard par rapport au rythme habituel',
  share_of_client: 'Part du couple dans le client',
  share_of_article: 'Part du couple dans l’article',
  client_amount: 'CA total client',
  client_n_orders: 'Commandes du client',
  client_active_months: 'Mois actifs client',
  client_days_since_last: 'Jours depuis la dernière commande du client',
  article_amount: 'CA de l’article',
  article_n_clients: 'Nombre de clients de l’article',
  article_repurchase_share: 'Part des clients qui rachètent',
  vendor_code: 'Vendeur',
  // --- article adoption model ---
  article_ca: 'CA article',
  article_n_buyers: 'Acheteurs de l’article',
  article_n_vendors: 'Vendeurs qui le placent',
  article_mean_ca_per_buyer: 'CA moyen par acheteur',
  client_ca: 'CA total client',
  client_n_months: 'Mois d’achat client',
  client_n_vendors: 'Vendeurs du client',
  client_n_families: 'Familles détenues',
  client_mean_ca_per_article: 'CA moyen par article',
  cosine_basket_sim: 'Ressemblance des paniers',
  knn_buyers: 'Acheteurs parmi les voisins',
  lift_weighted_overlap: 'Recouvrement pondéré par la spécificité',
  n_signature_held: 'Articles-repères détenus',
  share_signature_held: 'Part des articles-repères détenus',
  brand_n_held: 'Références détenues de la marque',
  brand_ca_share: 'Part de la marque',
  family_n_held: 'Références détenues de la famille',
  family_ca_share: 'Part de la famille',
  family: 'Famille',
  // --- shared by both ---
  client_n_articles: 'Largeur du panier client',
  enseigne: 'Marque',
}

export default function ModelsPage() {
  const { data, error, loading } = useFetch<Overview>(endpoints.overview)
  if (error) return <ErrorState error={error} />
  if (loading || !data) return <Loading label="Lecture des rapports de modèles" />

  return (
    <>
      <PageHeader
        eyebrow="Transparence du modèle"
        title="Pourquoi ces recommandations apparaissent-elles ?"
        description="Le dashboard ne demande pas de faire confiance à une boîte noire. Il montre la question de chaque modèle, sa mesure honnête et les faits qui influencent le classement."
      />

      <div className="model-explainer-grid">
        {data.models.map((model, index) => {
          const features = model.top_features.map(feature => ({
            ...feature,
            label: labels[feature.feature] ?? feature.feature.replaceAll('_', ' '),
          }))
          return (
            <section className="panel model-detail" key={model.name}>
              <div className="model-number">0{index + 1}</div>
              <span className="eyebrow">{model.name === 'repurchase' ? 'Modèle de rachat' : 'Modèle de prospection'}</span>
              <h2>{model.name === 'repurchase' ? 'Que va-t-il probablement reprendre ?' : 'Qui ressemble aux futurs adoptants ?'}</h2>
              <p>{model.explanation}</p>

              <div className="model-score-row">
                <div>
                  <strong>{formatPercent(model.headline_value)}</strong>
                  <span>{model.headline_metric}</span>
                </div>
                <div className="versus">vs</div>
                <div>
                  <strong>{formatPercent(model.comparison_value)}</strong>
                  <span>{model.comparison_label}</span>
                </div>
              </div>
              {model.auc !== null ? (
                <p className="muted-copy model-auc">
                  AUC sur la fenêtre de test : <strong>{model.auc.toFixed(3)}</strong>. Elle juge le
                  classement sur tous les couples, là où la mesure ci-dessus ne regarde que la
                  tête de liste — celle que le vendeur appelle vraiment.
                </p>
              ) : null}

              {model.name === 'repurchase' ? (
                <Formula><span>Historique du couple</span><strong>+</strong><span>profil client</span><strong>+</strong><span>popularité article</span></Formula>
              ) : (
                <Formula><span>panier</span><strong>+</strong><span>voisins</span><strong>+</strong><span>marque / famille</span></Formula>
              )}

              <h3>Ce qui influence le plus le classement</h3>
              {features.length ? (
                <div className="chart-wrap feature-chart">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={features} layout="vertical" margin={{ top: 4, right: 10, left: 18, bottom: 4 }}>
                      <CartesianGrid stroke="#ebe5db" horizontal={false} />
                      <XAxis type="number" hide />
                      <YAxis type="category" dataKey="label" width={145} tickLine={false} axisLine={false} tick={{ fontSize: 10 }} />
                      <Tooltip formatter={(value: number) => [value.toFixed(2), 'Importance']} contentStyle={chartTooltipStyle} />
                      <Bar dataKey="importance" fill={model.name === 'repurchase' ? '#0f5c4a' : '#e4572e'} radius={[0, 5, 5, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <p className="muted-copy">Les importances détaillées ne sont pas incluses dans ce jeu de démonstration.</p>
              )}
            </section>
          )
        })}
      </div>

      <SectionTitle index="03" title="Le protocole qui évite de tricher avec le futur" />
      <div className="timeline">
        <article><span>12 mois</span><strong>Observer</strong><p>Les caractéristiques ne sont construites qu’à partir de ce qui était déjà arrivé dans la fenêtre d’historique.</p></article>
        <div className="timeline-arrow">→</div>
        <article><span>Embargo</span><strong>Couper</strong><p>Une coupure sépare l’apprentissage de la mesure. Sans elle, un même achat pourrait servir à la fois de fait observé et de réponse à prédire.</p></article>
        <div className="timeline-arrow">→</div>
        <article className="test-year"><span>3 mois</span><strong>Mesurer</strong><p>L’achat observé dans les trois mois suivants sert d’étiquette, et l’opération est rejouée à chaque instantané mensuel.</p></article>
      </div>

      <div className="truth-grid">
        <section className="panel">
          <SectionTitle title="Ce que le score signifie" />
          <ul className="check-list">
            <li>Il permet de mettre les meilleures pistes en premier.</li>
            <li>Il combine plusieurs faits historiques plutôt qu’une seule règle.</li>
            <li>Il peut être comparé, sur la même fenêtre, au réflexe d’appeler d’abord les plus gros montants.</li>
          </ul>
        </section>
        <section className="panel">
          <SectionTitle title="Ce que le score ne signifie pas" />
          <ul className="cross-list">
            <li>Ce n’est pas une garantie d’achat ni de revenu.</li>
            <li>Il ne connaît ni le stock, ni le prix courant, ni la marge.</li>
            <li>Il ne sait pas quel jour appeler ni quelle quantité proposer.</li>
          </ul>
        </section>
      </div>

      <NoticePanel
        title="Règle d’affichage appliquée partout"
        items={[
          'Le modèle de rachat expose un score brut accompagné de la mention “estimation optimiste”.',
          'Le modèle de prospection expose un ordre et des preuves factuelles, jamais un faux pourcentage.',
          'Les filtres de stock, prix, compatibilité et autorisation restent des contrôles commerciaux séparés.',
        ]}
      />
    </>
  )
}
