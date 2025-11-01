import React from 'react'

export default function Footer() {
  return (
    <footer className="footer">
      <p>© {new Date().getFullYear()} Medical Report Interpreter</p>
      <p>Hybrid Rule + ML • Flask API • React + Vite</p>
    </footer>
  )
}
