import { NavLink, Outlet } from 'react-router-dom'

const navigation = [
  { to: '/', label: 'Vue d’ensemble', icon: '01' },
  { to: '/vendeurs', label: 'Vendeurs', icon: '02' },
  { to: '/experience-vendeur', label: 'Expérience vendeur', icon: '03' },
  { to: '/clients', label: 'Clients', icon: '04' },
  { to: '/marques', label: 'Analyse des marques', icon: '05' },
  { to: '/articles', label: 'Prospection article', icon: '06' },
  { to: '/modeles', label: 'Pourquoi le modèle', icon: '07' },
  { to: '/donnees', label: 'Données & limites', icon: '08' },
]

export default function Layout() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">S</div>
          <div>
            <strong>Sales Compass</strong>
            <span>Intelligence commerciale</span>
          </div>
        </div>

        <nav className="primary-nav" aria-label="Navigation principale">
          {navigation.map(item => (
            <NavLink key={item.to} to={item.to} end={item.to === '/'}>
              <span className="nav-index">{item.icon}</span>
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="privacy-note">
          <span className="privacy-dot" />
          <div>
            <strong>Données locales</strong>
            <p>Les noms clients et les montants restent sur cet ordinateur.</p>
          </div>
        </div>
      </aside>

      <main className="workspace">
        <header className="mobile-header">
          <div className="brand compact">
            <div className="brand-mark">S</div>
            <strong>Sales Compass</strong>
          </div>
          <span>Tableau de bord</span>
        </header>
        <div className="page"><Outlet /></div>
      </main>
    </div>
  )
}
