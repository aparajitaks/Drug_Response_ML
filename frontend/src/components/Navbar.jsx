import { Link, NavLink } from 'react-router-dom'

export default function Navbar() {
  const linkClass = ({ isActive }) =>
    `text-sm font-medium transition ${
      isActive ? 'text-indigo-600' : 'text-gray-600 hover:text-indigo-600'
    }`

  return (
    <header className="sticky top-0 z-20 border-b border-gray-200 bg-white/95 backdrop-blur">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
        <Link to="/" className="text-xl font-bold text-indigo-600">
          DrugIQ
        </Link>
        <div className="flex items-center gap-5">
          <NavLink to="/search" className={linkClass}>
            Search
          </NavLink>
          <NavLink to="/compare" className={linkClass}>
            Compare
          </NavLink>
          <NavLink to="/predict" className={linkClass}>
            Analyze
          </NavLink>
        </div>
      </nav>
    </header>
  )
}
