import React from 'react'
import { Link, NavLink } from 'react-router-dom'

export default function NavBar() {
  return (
    <nav className="navbar">
      <div className="nav-container">
        <Link to="/" className="nav-brand">🏥 Med Interpreter</Link>
        <ul className="nav-links">
          <li>
            <NavLink to="/" className={({isActive}) => 'nav-link' + (isActive ? ' active' : '')}>Home</NavLink>
          </li>
          <li>
            <NavLink to="/app" className={({isActive}) => 'nav-link' + (isActive ? ' active' : '')}>Demo</NavLink>
          </li>
          <li>
            <a className="nav-link" href="https://github.com/AnuprabhSingh/AI-Report-Analyser" target="_blank" rel="noreferrer">GitHub</a>
          </li>
          <li>
            <a className="nav-link" href="/frontend/README.md" target="_blank" rel="noreferrer">Docs</a>
          </li>
        </ul>
      </div>
    </nav>
  )
}
