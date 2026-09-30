import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { endpoints, formatMoney, formatNumber, type Overview } from '../api'
import {
  chartTooltipStyle,
  ErrorState,
  Loading,
  MetricCard,
  NoticePanel,
  PageHeader,
  SectionTitle,
} from '../components/ui'
import { useFetch } from '../useFetch'

export default function DataPage() {
  const { data, error, loading } = useFetch<Overview>(endpoints.overview)
  if (error) return <ErrorState error={error} />
  if (loading || !data) return <Loading label="Profilage du jeu de données" />

  return (
    <>
      <PageHeader
        eyebrow="Qualité & périmètre"
        title="Ce que les données permettent — et ce qu’elles interdisent."
        description="Le modèle est aussi bon que la question compatible avec l’extrait. Ici, une ligne représente du chiffre d’affaires mensuel, pas une commande."
      />

      <div className="metric-grid data-metrics">
        <MetricCard label="Lignes de faits" value={formatNumber(data.fact_rows)} detail="vendeur × client × article × mois" tone="pine" />
        <MetricCard label="Mois observés" value={formatNumber(data.observed_months)} detail={data.data_window} tone="clay" />
        <MetricCard label="Clients distincts" value={formatNumber(data.known_clients)} detail="les noms ne sont pas tous renseignés" />
        <MetricCard label="Articles distincts" value={formatNumber(data.catalog_articles)} detail="catalogue très long et très peu dense" tone="gold" />
      </div>

      <section className="panel chart-panel data-coverage-panel">
        <SectionTitle
          index="01"
          title="Couverture réellement disponible"
          description={`${formatNumber(data.observed_months)} mois consécutifs, sans trou de calendrier. La limite restante est la profondeur totale, pas les mois manquants.`}
        />
        <div className="chart-wrap data-chart">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.monthly_ca} margin={{ top: 12, right: 18, left: 4, bottom: 4 }}>
              <CartesianGrid stroke="#ebe5db" vertical={false} />
              <XAxis dataKey="month" tickLine={false} axisLine={false} tick={{ fontSize: 11 }} />
              <YAxis
                tickFormatter={(value: number) => formatMoney(value, true).replace(' TND', '')}
                tickLine={false}
                axisLine={false}
                width={58}
                tick={{ fontSize: 11 }}
              />
              <Tooltip formatter={(value: number) => [formatMoney(value), 'CA']} contentStyle={chartTooltipStyle} />
              <Bar dataKey="ca_tnd" fill="#0f5c4a" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>

      <SectionTitle index="02" title="Le dictionnaire de lecture" />
      <div className="definition-grid">
        <article><span>Une ligne</span><strong>Un montant mensuel agrégé</strong><p>Ce n’est ni une facture ni une commande. Plusieurs événements peuvent être additionnés.</p></article>
        <article><span>CA positif</span><strong>Un achat observé</strong><p>C’est la définition utilisée par les modèles pour dire qu’un couple client-article existe.</p></article>
        <article><span>CA négatif</span><strong>Retour ou avoir net</strong><p>L’extrait ne permet pas d’identifier la cause ni de relier le montant à une vente précise.</p></article>
        <article><span>Client vendeur</span><strong>Une relation plusieurs-à-plusieurs</strong><p>Un client peut apparaître sous plusieurs vendeurs. Le dashboard garde toujours le portefeuille sélectionné.</p></article>
      </div>

      <SectionTitle index="03" title="Informations absentes" description="Aucun écran ne doit inventer ces éléments." />
      <div className="missing-grid">
        {[
          ['Dates exactes', 'Impossible de recommander “appelez mardi” ou de calculer un cycle en jours.'],
          ['Quantités', 'Impossible de dire combien d’unités proposer.'],
          ['Prix et marge', 'Le CA historique n’est pas le prix courant ni la rentabilité.'],
          ['Stock', 'Une bonne recommandation peut être indisponible commercialement.'],
          ['Compatibilité', 'Le véhicule ou la référence technique doivent être contrôlés ailleurs.'],
          ['Historique d’appels', 'La causalité entre appel et achat n’est pas encore mesurée.'],
        ].map(([title, text]) => (
          <article key={title}><span>×</span><div><strong>{title}</strong><p>{text}</p></div></article>
        ))}
      </div>

      <section className="panel source-card">
        <SectionTitle index="04" title="Source actuelle" />
        <p>{data.generated_from}.</p>
        <div>
          <span>Fenêtres</span><strong>{data.data_window}</strong>
          <span>CA total observé</span><strong>{formatMoney(data.total_ca_tnd)}</strong>
        </div>
      </section>

      <NoticePanel title="Questions à confirmer avec l’entreprise" items={data.notices} />
    </>
  )
}
