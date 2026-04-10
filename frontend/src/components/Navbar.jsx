import { Link, NavLink } from 'react-router-dom'

const navItemClass = ({ isActive }) =>
  `text-sm font-medium ${isActive ? 'text-indigo-600' : 'text-gray-600 hover:text-indigo-600'}`

export default function Navbar() {
  return (
    <header className="sticky top-0 z-20 border-b border-gray-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link to="/" className="text-xl font-bold text-indigo-600">
          DrugIQ
        </Link>
        <nav className="flex items-center gap-6">
          <NavLink to="/search" className={navItemClass}>
            Search
          </NavLink>
          <NavLink to="/compare" className={navItemClass}>
            Compare
          </NavLink>
          <NavLink to="/predict" className={navItemClass}>
            Analyze
          </NavLink>
        </nav>
      </div>
    </header>
  )
}
