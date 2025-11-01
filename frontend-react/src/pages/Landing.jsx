import React from 'react'
import { Link } from 'react-router-dom'

export default function Landing() {
  return (
    <div className="landing">
      <section className="hero">
        <div className="hero-grid">
          <div>
            <h1>AI-Powered Echocardiography Analysis</h1>
            <p className="hero-subtitle">Extract measurements from PDF reports, apply validated clinical rules and optional ML, and visualize results with insightful dashboards.</p>
            <div className="cta-group">
              <Link to="/app" className="btn-primary">🚀 Open Demo</Link>
              <a href="https://github.com/AnuprabhSingh/AI-Report-Analyser" target="_blank" rel="noreferrer" className="btn-secondary">⭐ Star on GitHub</a>
            </div>
          </div>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-number">3</span>
              <div className="stat-label">Core Categories</div>
            </div>
            <div className="stat-item">
              <span className="stat-number">+8</span>
              <div className="stat-label">Key Parameters</div>
            </div>
            <div className="stat-item">
              <span className="stat-number">CV</span>
              <div className="stat-label">Cross-Validation</div>
            </div>
          </div>
        </div>
      </section>

      <section className="section">
        <h2 className="section-title">Why you'll love it</h2>
        <p className="section-subtitle">Designed for clarity, speed, and trust—backed by rules with optional ML.</p>
        <div className="feature-grid">
          <div className="glass-card">
            <div className="feature-icon">🤖</div>
            <h3>Hybrid Intelligence</h3>
            <p>Deterministic clinical rule engine with optional machine learning overlay for nuanced decisions.</p>
          </div>
          <div className="glass-card">
            <div className="feature-icon">📄</div>
            <h3>PDF Extraction</h3>
            <p>Robust parsing for common echo report formats, including tables and free text measurements.</p>
          </div>
          <div className="glass-card">
            <div className="feature-icon">📚</div>
            <h3>Batch Processing</h3>
            <p>Analyze entire folders of reports at once; get aggregate distributions and per-file summaries.</p>
          </div>
          <div className="glass-card">
            <div className="feature-icon">📊</div>
            <h3>Model Comparison</h3>
            <p>Compare algorithms with accuracy, precision, recall, F1, timing, and confusion matrices.</p>
          </div>
        </div>
      </section>

      <section className="section">
        <h2 className="section-title">How it works</h2>
        <div className="steps">
          <div className="step">
            <div className="step-num">1</div>
            <div className="step-content">
              <div className="step-title">Upload or Enter</div>
              <div className="step-text">Upload a PDF report or manually input measurements.</div>
            </div>
          </div>
          <div className="step">
            <div className="step-num">2</div>
            <div className="step-content">
              <div className="step-title">Extract & Normalize</div>
              <div className="step-text">We parse relevant metrics and normalize values where needed.</div>
            </div>
          </div>
          <div className="step">
            <div className="step-num">3</div>
            <div className="step-content">
              <div className="step-title">Interpret</div>
              <div className="step-text">Rules apply evidence-based thresholds; ML (optional) refines categorization.</div>
            </div>
          </div>
          <div className="step">
            <div className="step-num">4</div>
            <div className="step-content">
              <div className="step-title">Visualize</div>
              <div className="step-text">Interactive charts and JSON outputs for transparency and analysis.</div>
            </div>
          </div>
        </div>
        <div style={{ textAlign: 'center', marginTop: 20 }}>
          <Link to="/app" className="btn-primary">Try the Demo →</Link>
        </div>
      </section>
    </div>
  )
}
