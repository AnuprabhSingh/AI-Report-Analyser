import React from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App from './App.jsx'
import Landing from './pages/Landing.jsx'
import NavBar from './components/NavBar.jsx'
import Footer from './components/Footer.jsx'
import './styles.css'

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <NavBar />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/app" element={<App />} />
        <Route path="*" element={<Landing />} />
      </Routes>
      <Footer />
    </BrowserRouter>
  </React.StrictMode>
)
