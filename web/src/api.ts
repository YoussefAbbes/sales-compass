const API = '/api/v1'

export async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal, headers: { Accept: 'application/json' } })
  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      // Keep the status when the backend did not return JSON.
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

// Mirrors getJSON deliberately, including sending no X-API-Key: the dashboard has no
// mechanism for configuring one, and a write helper that authenticates while the read
// helper does not would be inconsistent and untested.
export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(path, {
    method: 'POST',
    headers: { Accept: 'application/json', 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try {
      const parsed = (await response.json()) as { detail?: string }
      if (parsed.detail) detail = parsed.detail
    } catch {
      // Keep the status when the backend did not return JSON.
    }
    throw new Error(detail)
  }
  return response.json() as Promise<T>
}

export type FeedbackAction =
  | 'CONTACTED' | 'ORDER' | 'QUOTE' | 'SNOOZE' | 'UNREACHABLE' | 'CLOSED' | 'DISMISS'

export type FeedbackEvent = {
  event_id: string
  vendor_code: string
  client_id: string
  action: FeedbackAction
  note: string
  recorded_at: string
  score_snapshot: string | null
}

export const endpoints = {
  overview: `${API}/dashboard/overview`,
  vendors: `${API}/vendors`,
  vendorDashboard: (vendor: string) => `${API}/vendors/${encodeURIComponent(vendor)}/dashboard`,
  client: (vendor: string, client: string) =>
    `${API}/vendors/${encodeURIComponent(vendor)}/clients/${encodeURIComponent(client)}?per_client=50`,
  brands: (query = '') => `${API}/brands?query=${encodeURIComponent(query)}&limit=300`,
  brandDetail: (brand: string) => `${API}/brands/detail?brand=${encodeURIComponent(brand)}&limit=10`,
  articles: (query: string) => `${API}/articles?query=${encodeURIComponent(query)}&limit=30`,
  recommendations: (vendor: string, article: string) =>
    `${API}/vendors/${encodeURIComponent(vendor)}/articles/${encodeURIComponent(article)}/recommendations?limit=200`,
  meta: `${API}/meta`,
  feedback: `${API}/feedback`,
  vendorFeedback: (vendor: string) => `${API}/vendors/${encodeURIComponent(vendor)}/feedback`,
}

export const formatNumber = (value: number, digits = 0) =>
  value.toLocaleString('fr-FR', { maximumFractionDigits: digits })

export const formatMoney = (value: number, compact = false) => {
  if (compact) {
    return new Intl.NumberFormat('fr-FR', {
      notation: 'compact',
      maximumFractionDigits: 1,
    }).format(value) + ' TND'
  }
  return `${formatNumber(value)} TND`
}

// Mirrors _repurchase_reason server-side: one decimal is real precision at 2x and
// false precision at 21x, where the elapsed days already carry the message.
export const formatRatio = (value: number | null) => {
  if (value === null) return '—'
  const digits = value >= 10 ? 0 : 1
  return `${value.toLocaleString('fr-FR', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  })}×`
}

export const formatPercent = (value: number | null, digits = 1) =>
  value === null ? '—' : `${(value * 100).toFixed(digits)} %`

export const statusLabel = (status: string) => ({
  ACTIVE: 'Actif, articles en attente',
  DECLINING: 'En baisse',
  INACTIVE_2026: 'Inactif en 2026',
}[status] ?? status)

export type MonthlyCA = { month: string; ca_tnd: number }
export type StatusBreakdown = { status: string; count: number; tnd_at_stake: number }
export type FeatureImportance = { feature: string; importance: number }

export type BrandSummary = {
  brand: string
  rank: number
  ca_tnd: number
  share_of_scope_ca: number
  clients: number
  articles: number
  purchase_months: number
  ca_2025_tnd: number
  ca_2026_tnd: number
  evolution_pct: number | null
}

export type BrandRankItem = {
  rank: number
  code: string
  name: string
  ca_tnd: number
  share_of_brand_ca: number
  clients: number | null
  articles: number | null
}

export type Vendor = {
  vendor_code: string
  vendor_name: string
  queued_clients: number
  tnd_at_stake: number
}

export type DashboardClient = {
  rank: number
  client_id: string
  client_name: string
  status: string
  priority_score: number
  expected_value_tnd: number
  lost_revenue_tnd: number
  key_amount_tnd: number
  amount_explanation: string
  last_activity: string | null
  summary: string
  top_article: string | null
  why_recommended: string[]
}

export type ModelSummary = {
  name: 'repurchase' | 'article_prospect'
  headline_metric: string
  headline_value: number | null
  comparison_label: string
  comparison_value: number | null
  auc: number | null
  explanation: string
  top_features: FeatureImportance[]
}

export type Overview = {
  data_window: string
  generated_from: string
  total_ca_tnd: number
  fact_rows: number
  observed_months: number
  catalog_articles: number
  known_clients: number
  vendors: number
  queued_assignments: number
  tnd_at_stake: number
  monthly_ca: MonthlyCA[]
  status_breakdown: StatusBreakdown[]
  vendor_comparison: Vendor[]
  known_brand_ca_share: number
  top_brands: BrandSummary[]
  top_clients: DashboardClient[]
  models: ModelSummary[]
  notices: string[]
}

export type VendorDashboard = {
  vendor_code: string
  vendor_name: string
  queued_clients: number
  tnd_at_stake: number
  expected_value_tnd: number
  lost_revenue_tnd: number
  monthly_ca: MonthlyCA[]
  status_breakdown: StatusBreakdown[]
  known_brand_ca_share: number
  brand_concentration_top5: number
  brands: BrandSummary[]
  clients: DashboardClient[]
  notices: string[]
}

export type Opportunity = {
  rank: number
  article_id: string
  brand: string | null
  designation: string
  repurchase_score_raw: number
  previous_ca_tnd: number
  expected_value_tnd: number
  reason: string
}

export type ClientDashboard = {
  vendor_code: string
  vendor_name: string
  client_id: string
  client_name: string
  rank: number
  status: string
  priority_score: number
  expected_value_tnd: number
  lost_revenue_tnd: number
  key_amount_tnd: number
  amount_explanation: string
  last_activity: string | null
  summary: string
  why_recommended: string[]
  monthly_ca: MonthlyCA[]
  yearly_ca_tnd: Record<string, number>
  known_brand_ca_share: number
  brand_concentration_top3: number
  brands: BrandSummary[]
  opportunities: Opportunity[]
  cross_sell_suggestion: string | null
  notices: string[]
}

export type Article = {
  article_id: string
  brand: string | null
  designation: string
  label: string
}

export type ArticleSearch = { query: string; items: Article[] }

export type BrandList = {
  total_brands: number
  known_brand_ca_share: number
  items: BrandSummary[]
  notices: string[]
}

export type BrandDashboard = {
  brand: string
  ca_tnd: number
  share_of_known_brand_ca: number
  clients: number
  vendors: number
  articles: number
  purchase_months: number
  monthly_ca: MonthlyCA[]
  yearly_ca_tnd: Record<string, number>
  top_vendors: BrandRankItem[]
  top_clients: BrandRankItem[]
  top_articles: BrandRankItem[]
  notices: string[]
}

export type ProspectSignals = {
  similar_buyers_out_of_20: number
  signature_articles_owned: number
  signature_articles_total: number
  same_brand_articles_owned: number
  same_family_articles_owned: number
}

export type ArticleRecommendations = {
  vendor_code: string
  article: Article
  repurchases: {
    model: 'repurchase'
    model_version: string | null
    data_window: string
    score_is_calibrated: false
    items: Array<{
      rank: number
      client_id: string
      client_name: string
      repurchase_score_raw: number
      previous_ca_tnd: number
      expected_value_tnd: number
      signals: {
        n_orders: number
        days_since_last: number
        median_interval_days: number | null
        overdue_ratio: number | null
      }
      reason: string
    }>
    notices: string[]
  }
  prospects: {
    model: 'article_prospect'
    model_version: string | null
    source: 'model' | 'heuristic'
    data_window: string
    probability_available: false
    current_window_buyers: number
    minimum_buyers_required: number
    eligible_candidates: number
    empty_reason: string | null
    items: Array<{
      rank: number
      client_id: string
      client_name: string
      client_ca_tnd: number
      signals: ProspectSignals
      reason: string
    }>
    notices: string[]
  }
}
