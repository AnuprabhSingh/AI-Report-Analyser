import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Chart, BarElement, BarController, CategoryScale, LinearScale, Legend, LineElement, LineController, PointElement, Title, Tooltip } from 'chart.js'

Chart.register(BarElement, BarController, CategoryScale, LinearScale, Legend, LineElement, LineController, PointElement, Title, Tooltip)

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000'

function Alert({ type = 'info', message = '', hidden = false }) {
  if (hidden || !message) return null
  const cls = type === 'error' ? 'alert error' : type === 'success' ? 'alert success' : 'alert info'
  return <div className={cls}>{message}</div>
}

function Tabs({ active, onChange }) {
  const tabs = [
    { key: 'upload', label: '📤 Upload PDF' },
    { key: 'manual', label: '⌨️ Manual Entry' },
    { key: 'batch', label: '📚 Batch Upload' },
    { key: 'metrics', label: '📊 Model Comparison' }
  ]
  return (
    <div className="tabs">
      {tabs.map(t => (
        <button key={t.key} className={"tab" + (active === t.key ? ' active' : '')} onClick={() => onChange(t.key)}>
          {t.label}
        </button>
      ))}
    </div>
  )
}

function UploadTab({ onResults }) {
  const [loading, setLoading] = useState(false)
  const [alert, setAlert] = useState({ type: 'info', message: '', hidden: true })
  const fileInput = useRef()
  const uploadArea = useRef()

  useEffect(() => {
    const area = uploadArea.current
    if (!area) return
    const onDragOver = e => { e.preventDefault(); area.classList.add('dragover') }
    const onDragLeave = () => area.classList.remove('dragover')
    const onDrop = e => {
      e.preventDefault(); area.classList.remove('dragover')
      const files = e.dataTransfer.files
      if (files.length > 0 && (files[0].type === 'application/pdf' || files[0].name.toLowerCase().endsWith('.pdf'))) {
        uploadFile(files[0])
      } else {
        setAlert({ type: 'error', message: 'Please upload a PDF file', hidden: false })
      }
    }
    area.addEventListener('dragover', onDragOver)
    area.addEventListener('dragleave', onDragLeave)
    area.addEventListener('drop', onDrop)
    return () => {
      area.removeEventListener('dragover', onDragOver)
      area.removeEventListener('dragleave', onDragLeave)
      area.removeEventListener('drop', onDrop)
    }
  }, [])

  async function uploadFile(file) {
    setLoading(true); setAlert(a => ({ ...a, hidden: true }))
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_BASE}/api/interpret`, { method: 'POST', body: formData })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Failed to process report')
      onResults(data)
      setAlert({ type: 'success', message: '✅ Report processed successfully!', hidden: false })
    } catch (e) {
      setAlert({ type: 'error', message: `❌ ${e.message}`, hidden: false })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className={"tab-content active"}>
      <div className="upload-area" ref={uploadArea} onClick={() => fileInput.current?.click()}>
        <div className="upload-icon">📄</div>
        <div className="upload-text">Click to upload or drag and drop</div>
        <div className="upload-hint">PDF medical reports (Max 16MB)</div>
        <input ref={fileInput} type="file" accept=".pdf" onChange={e => e.target.files[0] && uploadFile(e.target.files[0])} />
      </div>
      {loading && (
        <div className="loading active">
          <div className="spinner"></div>
          <p>Processing your report...</p>
        </div>
      )}
      <Alert {...alert} />
    </div>
  )
}

function ManualTab({ onResults }) {
  const [loading, setLoading] = useState(false)
  const [alert, setAlert] = useState({ type: 'info', message: '', hidden: true })

  async function onSubmit(e) {
    e.preventDefault()
    const form = e.currentTarget
    const data = {
      patient: { age: parseInt(form.age.value || '0', 10), sex: form.sex.value },
      measurements: {}
    }
    const mapping = { EF: 'ef', LVID_D: 'lvid_d', LVID_S: 'lvid_s', IVS_D: 'ivs_d', LVPW_D: 'lvpw_d', LA_DIMENSION: 'la_dim', MV_E_A: 'mv_ea', FS: 'fs' }
    Object.entries(mapping).forEach(([k, id]) => { const v = form[id].value; if (v) data.measurements[k] = parseFloat(v) })
    if (Object.keys(data.measurements).length === 0) { setAlert({ type: 'error', message: '❌ Please enter at least one measurement', hidden: false }); return }
    setLoading(true); setAlert(a => ({ ...a, hidden: true }))
    try {
      const res = await fetch(`${API_BASE}/api/interpret/json`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) })
      const json = await res.json()
      if (!res.ok) throw new Error(json.error || 'Failed to analyze')
      onResults(json)
      setAlert({ type: 'success', message: '✅ Analysis completed successfully!', hidden: false })
    } catch (e) {
      setAlert({ type: 'error', message: `❌ ${e.message}`, hidden: false })
    } finally { setLoading(false) }
  }

  return (
    <div className="tab-content active">
      <form id="manualForm" onSubmit={onSubmit}>
        <h3 style={{ marginBottom: 20, color: '#333' }}>Patient Information</h3>
        <div className="grid">
          <div className="form-group">
            <label>Age</label>
            <input type="number" name="age" placeholder="45" min="1" max="120" required />
          </div>
          <div className="form-group">
            <label>Sex</label>
            <select name="sex" required>
              <option value="">Select...</option>
              <option value="M">Male</option>
              <option value="F">Female</option>
            </select>
          </div>
        </div>
        <h3 style={{ margin: '30px 0 20px', color: '#333' }}>Measurements</h3>
        <div className="grid">
          {[
            ['EF (Ejection Fraction) %', 'ef', '64.8', 0.1],
            ['LVIDd (cm)', 'lvid_d', '4.65', 0.01],
            ['LVIDs (cm)', 'lvid_s', '3.00', 0.01],
            ['IVSd (cm)', 'ivs_d', '1.13', 0.01],
            ['LVPWd (cm)', 'lvpw_d', '1.02', 0.01],
            ['LA Dimension (cm)', 'la_dim', '2.38', 0.01],
            ['MV E/A Ratio', 'mv_ea', '1.75', 0.01],
            ['FS (Fractional Shortening) %', 'fs', '35.4', 0.1]
          ].map(([label, name, ph, step]) => (
            <div className="form-group" key={name}>
              <label>{label}</label>
              <input type="number" name={name} placeholder={ph} step={step} />
            </div>
          ))}
        </div>
        <div style={{ marginTop: 30, textAlign: 'center' }}>
          <button type="submit" className="btn btn-primary">🔍 Analyze Measurements</button>
        </div>
      </form>
      <div className={"loading" + (loading ? ' active' : '')}>
        <div className="spinner" />
        <p>Analyzing measurements...</p>
      </div>
      <Alert {...alert} />
    </div>
  )
}

function useChart(canvasRef, buildConfig, deps) {
  useEffect(() => {
    if (!canvasRef.current) return
    const ctx = canvasRef.current.getContext('2d')
    const cfg = buildConfig()
    const chart = new Chart(ctx, cfg)
    return () => chart.destroy()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
}

function BatchTab() {
  const [loading, setLoading] = useState(false)
  const [alert, setAlert] = useState({ type: 'info', message: '', hidden: true })
  const [results, setResults] = useState([])
  const [dist, setDist] = useState({ ages: [], efs: [], la_dims: [], cats: { LV_HYPERTROPHY: {}, LA_SIZE: {}, DIASTOLIC_FUNCTION: {} } })

  const ageCanvas = useRef(); const efCanvas = useRef(); const laCanvas = useRef();

  function handleFiles(files) {
    const form = new FormData()
    Array.from(files).forEach(file => {
      if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
        form.append('files', file)
      }
    })
    uploadBatch(form)
  }

  async function uploadBatch(formData) {
    setLoading(true); setAlert(a => ({ ...a, hidden: true }))
    try {
      const res = await fetch(`${API_BASE}/api/batch`, { method: 'POST', body: formData })
      const json = await res.json()
      if (!res.ok) throw new Error(json.error || 'Failed to process batch')
      setResults(json.results || [])
      setAlert({ type: 'success', message: `✅ Processed ${json.success} files successfully! ${json.failed > 0 ? `(${json.failed} failed)` : ''}`, hidden: false })
      // Build distributions
      const ages = []; const efs = []; const la_dims = []
      const cats = { LV_HYPERTROPHY: {}, LA_SIZE: {}, DIASTOLIC_FUNCTION: {} }
      for (const r of json.results || []) {
        if (r.patient?.age) ages.push(r.patient.age)
        if (r.measurements?.EF) efs.push(r.measurements.EF)
        if (r.measurements?.LA_DIMENSION) la_dims.push(r.measurements.LA_DIMENSION)
        const labels = r.class_labels
        if (labels) {
          Object.keys(cats).forEach(c => { const lab = labels[c] || 'Unknown'; cats[c][lab] = (cats[c][lab] || 0) + 1 })
        } else if (r.interpretations) {
          Object.keys(cats).forEach(c => { const val = r.interpretations[c]; const m = (val||'').match(/\b(Normal|Mild|Moderate|Severe|Enlarged|Abnormal|Dilated|None)\b/i); const lab = m ? m[0] : 'Unknown'; cats[c][lab] = (cats[c][lab] || 0) + 1 })
        }
      }
      setDist({ ages, efs, la_dims, cats })
    } catch (e) {
      setAlert({ type: 'error', message: `❌ ${e.message}` , hidden: false })
    } finally { setLoading(false) }
  }

  useChart(ageCanvas, () => ({
    type: 'bar', data: { labels: dist.ages.map((_, i) => i + 1), datasets: [{ label: 'Age', data: dist.ages, backgroundColor: '#667eea', borderRadius: 6, maxBarThickness: 40, barPercentage: 0.7, categoryPercentage: 0.6 }] },
    options: { responsive: true, maintainAspectRatio: true, aspectRatio: 1.8, plugins: { legend: { position: 'bottom' } }, scales: { x: { title: { display: true, text: 'Sample' }, ticks: { maxRotation: 0, autoSkip: true } }, y: { title: { display: true, text: 'Age' }, beginAtZero: true } } }
  }), [JSON.stringify(dist.ages)])

  useChart(efCanvas, () => ({
    type: 'bar', data: { labels: dist.efs.map((_, i) => i + 1), datasets: [{ label: 'EF (%)', data: dist.efs, backgroundColor: '#764ba2', borderRadius: 6, maxBarThickness: 40, barPercentage: 0.7, categoryPercentage: 0.6 }] },
    options: { responsive: true, maintainAspectRatio: true, aspectRatio: 1.8, plugins: { legend: { position: 'bottom' } }, scales: { x: { title: { display: true, text: 'Sample' }, ticks: { maxRotation: 0, autoSkip: true } }, y: { title: { display: true, text: 'EF (%)' }, beginAtZero: true } } }
  }), [JSON.stringify(dist.efs)])

  useChart(laCanvas, () => ({
    type: 'bar', data: { labels: dist.la_dims.map((_, i) => i + 1), datasets: [{ label: 'LA Dimension (cm)', data: dist.la_dims, backgroundColor: '#4caf50', borderRadius: 6, maxBarThickness: 40, barPercentage: 0.7, categoryPercentage: 0.6 }] },
    options: { responsive: true, maintainAspectRatio: true, aspectRatio: 1.8, plugins: { legend: { position: 'bottom' } }, scales: { x: { title: { display: true, text: 'Sample' }, ticks: { maxRotation: 0, autoSkip: true } }, y: { title: { display: true, text: 'LA Dimension (cm)' }, beginAtZero: true } } }
  }), [JSON.stringify(dist.la_dims)])

  return (
    <div className="tab-content active">
      <div className="upload-area" onClick={() => document.getElementById('batchFiles').click()}>
        <div className="upload-icon">📚</div>
        <div className="upload-text">Click to upload multiple PDFs</div>
        <div className="upload-hint">Select multiple PDF files to process in batch</div>
        <input id="batchFiles" type="file" accept=".pdf" multiple onChange={e => e.target.files && handleFiles(e.target.files)} />
      </div>
      <div className={"loading" + (loading ? ' active' : '')}>
        <div className="spinner"></div>
        <p>Processing batch reports...</p>
      </div>
      <Alert {...alert} />

      {results.length > 0 && (
        <div id="batchResults" style={{ marginTop: 30 }}>
          <h3 style={{ marginBottom: 20 }}>Batch Processing Results</h3>
          <div id="batchResultsList">
            {results.map((r, i) => (
              <div key={i} className="result-card" style={{ marginBottom: 15 }}>
                <h4 style={{ marginBottom: 10 }}>{i + 1}. {r.file_name}</h4>
                <div><strong>Patient:</strong> Age {r.patient?.age || 'N/A'}, Sex {r.patient?.sex || 'N/A'}</div>
                <div><strong>Measurements:</strong> {Object.keys(r.measurements || {}).length} parameters extracted</div>
                <div><strong>Status:</strong> <span className="status-badge status-normal">{r.status || 'processed'}</span></div>
              </div>
            ))}
          </div>
          <div id="batchCharts" style={{ marginTop: 40 }}>
            {dist.ages.length > 0 && (<>
              <h4>Age Distribution</h4><canvas ref={ageCanvas} className="chart-canvas" />
            </>)}
            {dist.efs.length > 0 && (<>
              <h4>Ejection Fraction (EF) Distribution</h4><canvas ref={efCanvas} className="chart-canvas" />
            </>)}
            {dist.la_dims.length > 0 && (<>
              <h4>LA Dimension Distribution</h4><canvas ref={laCanvas} className="chart-canvas" />
            </>)}
            {Object.entries(dist.cats).map(([cat, vals]) => {
              const labels = Object.keys(vals)
              const values = labels.map(l => vals[l])
              if (labels.length === 0) return null
              return (
                <CatBar key={cat} title={`${cat} Distribution`} labels={labels} data={values} color="#ff9800" />
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

function CatBar({ title, labels, data, color }) {
  const ref = useRef()
  useChart(ref, () => ({ type: 'bar', data: { labels, datasets: [{ label: title, data, backgroundColor: color, barPercentage: 0.7, categoryPercentage: 0.6 }] }, options: { responsive: true, maintainAspectRatio: true, aspectRatio: 1.8, scales: { y: { title: { display: true, text: 'Count' }, beginAtZero: true } } } }), [JSON.stringify(labels), JSON.stringify(data)])
  return (<>
    <h4>{title}</h4>
    <canvas ref={ref} className="chart-canvas" />
  </>)
}

function LineChart({ title, labels, data, color = '#4f46e5', yLabel = 'Prediction' }) {
  const ref = useRef()
  useChart(ref, () => ({
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: title,
        data,
        borderColor: color,
        backgroundColor: 'rgba(79, 70, 229, 0.15)',
        tension: 0.35,
        pointRadius: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1.8,
      plugins: { legend: { position: 'bottom' } },
      scales: {
        x: { ticks: { maxRotation: 0, autoSkip: true } },
        y: { beginAtZero: false, title: { display: true, text: yLabel } }
      }
    }
  }), [JSON.stringify(labels), JSON.stringify(data), title, color, yLabel])
  return (
    <>
      <h4>{title}</h4>
      <canvas ref={ref} className="chart-canvas" />
    </>
  )
}

function SeverityGrader({ severity_grading, measurements, interpretations }) {
  if (!severity_grading || !severity_grading.grades) {
    // Fallback to old text-based parsing if no severity_grading data
    return <SeverityGraderLegacy measurements={measurements} interpretations={interpretations} />
  }

  const grades = severity_grading.grades || {}
  const severity_summary = severity_grading.severity_summary || {}
  
  const getSeverityInfo = (grade) => {
    let label = 'Unknown'
    let color = '#9ca3af'
    let numericGrade = 0
    
    // Parse grade for numeric value
    if (typeof grade === 'number') {
      numericGrade = grade
    } else if (grade?.numeric_grade !== undefined) {
      numericGrade = grade.numeric_grade
    } else if (typeof grade === 'string') {
      if (grade.includes('Indeterminate')) {
        numericGrade = 0
        label = 'Insufficient data'
        color = '#9ca3af'
      } else if (grade.includes('Grade 3') || grade.includes('Severe')) {
        numericGrade = 3
        label = 'Severe'
        color = '#ef4444'
      } else if (grade.includes('Grade 2') || grade.includes('Moderate')) {
        numericGrade = 2
        label = 'Moderate'
        color = '#f59e0b'
      } else if (grade.includes('Grade 1') || grade.includes('Mild')) {
        numericGrade = 1
        label = 'Mild'
        color = '#eab308'
      } else if (grade.includes('Normal') || grade === 'Normal') {
        numericGrade = 0
        label = 'Normal'
        color = '#10b981'
      }
    }
    
    return { numericGrade, label: label !== 'Unknown' ? label : (grade?.grade || 'Normal'), color }
  }

  const diastolic_info = getSeverityInfo(grades.diastolic_dysfunction?.grade)
  const lvh_info = getSeverityInfo(grades.lvh?.grade)
  const systolic_info = getSeverityInfo(grades.systolic_function?.grade)
  const diastolic_numeric = grades.diastolic_dysfunction?.numeric_grade ?? diastolic_info.numericGrade ?? 0
  const lvh_numeric = grades.lvh?.numeric_grade ?? lvh_info.numericGrade ?? 0
  const systolic_numeric = grades.systolic_function?.numeric_grade ?? systolic_info.numericGrade ?? 0
  
  return (
    <div className="result-card" style={{ marginTop: 24 }}>
      <div className="result-title">📊 Disease Severity Grading</div>
      <div className="severity-grid">
        {/* Diastolic Function */}
        {grades.diastolic_dysfunction && (
          <div className="severity-item" style={{ borderColor: diastolic_info.color }}>
            <div className="severity-header">
              <span className="severity-name">Diastolic Function</span>
              <span className="severity-grade" style={{ backgroundColor: diastolic_info.color }}>Grade {diastolic_numeric}</span>
            </div>
            <div className="severity-label" style={{ color: diastolic_info.color }}>{diastolic_info.label}</div>
            <div className="severity-bar">
              <div className="severity-bar-fill" style={{ width: `${(diastolic_numeric / 3) * 100}%`, backgroundColor: diastolic_info.color }} />
            </div>
            <div style={{ fontSize: '0.8rem', color: '#666', marginTop: 8 }}>
              Confidence: {((grades.diastolic_dysfunction.confidence ?? 0) * 100).toFixed(0)}%
            </div>
          </div>
        )}
        
        {/* LVH */}
        {grades.lvh && (
          <div className="severity-item" style={{ borderColor: lvh_info.color }}>
            <div className="severity-header">
              <span className="severity-name">LV Hypertrophy</span>
              <span className="severity-grade" style={{ backgroundColor: lvh_info.color }}>Grade {lvh_numeric}</span>
            </div>
            <div className="severity-label" style={{ color: lvh_info.color }}>{lvh_info.label}</div>
            {grades.lvh.geometry && <div style={{ fontSize: '0.8rem', color: '#666', marginTop: 4 }}>{grades.lvh.geometry}</div>}
            <div className="severity-bar">
              <div className="severity-bar-fill" style={{ width: `${(lvh_numeric / 3) * 100}%`, backgroundColor: lvh_info.color }} />
            </div>
            <div style={{ fontSize: '0.8rem', color: '#666', marginTop: 8 }}>
              Confidence: {((grades.lvh.confidence ?? 0) * 100).toFixed(0)}%
            </div>
          </div>
        )}
        
        {/* Systolic Function */}
        {grades.systolic_function && (
          <div className="severity-item" style={{ borderColor: systolic_info.color }}>
            <div className="severity-header">
              <span className="severity-name">Systolic Function</span>
              <span className="severity-grade" style={{ backgroundColor: systolic_info.color }}>Grade {systolic_numeric}</span>
            </div>
            <div className="severity-label" style={{ color: systolic_info.color }}>{systolic_info.label}</div>
            <div className="severity-bar">
              <div className="severity-bar-fill" style={{ width: `${(systolic_numeric / 3) * 100}%`, backgroundColor: systolic_info.color }} />
            </div>
            <div style={{ fontSize: '0.8rem', color: '#666', marginTop: 8 }}>
              Confidence: {((grades.systolic_function.confidence ?? 0) * 100).toFixed(0)}%
            </div>
          </div>
        )}
      </div>
      
      {/* Overall Summary */}
      {severity_summary.overall_score !== undefined && (
        <div style={{ marginTop: 20, padding: 15, backgroundColor: '#f3f4f6', borderRadius: 8 }}>
          <div style={{ fontSize: '0.9rem', color: '#666', marginBottom: 8 }}>
            <strong>Overall Severity Score:</strong> {severity_summary.overall_score.toFixed(1)}/10 - <span style={{ color: severity_summary.severity_level === 'Severe' ? '#ef4444' : severity_summary.severity_level === 'Moderate' ? '#f59e0b' : severity_summary.severity_level === 'Mild' ? '#eab308' : '#10b981', fontWeight: 600 }}>{severity_summary.severity_level}</span>
          </div>
          {severity_summary.primary_concerns && severity_summary.primary_concerns.length > 0 && (
            <div style={{ fontSize: '0.9rem', color: '#666' }}>
              <strong>Primary Concerns:</strong>
              <ul style={{ marginTop: 8, paddingLeft: 20 }}>
                {severity_summary.primary_concerns.map((concern, i) => (
                  <li key={i} style={{ color: '#555' }}>{concern}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function SeverityGraderLegacy({ measurements, interpretations }) {
  const determineSeverity = (category, value, interpretation) => {
    const text = (interpretation || '').toLowerCase()
    
    // LV Hypertrophy severity
    if (category === 'Interventricular Septum' || category === 'LV_HYPERTROPHY') {
      if (text.includes('severe')) return { grade: 3, label: 'Severe', color: '#ef4444' }
      if (text.includes('moderate')) return { grade: 2, label: 'Moderate', color: '#f59e0b' }
      if (text.includes('mild')) return { grade: 1, label: 'Mild', color: '#eab308' }
      return { grade: 0, label: 'Normal', color: '#10b981' }
    }
    
    // Diastolic Dysfunction severity
    if (category === 'Diastolic Function' || category === 'DIASTOLIC_FUNCTION') {
      if (text.includes('restrictive')) return { grade: 3, label: 'Grade 3 (Restrictive)', color: '#ef4444' }
      if (text.includes('pseudonormal')) return { grade: 2, label: 'Grade 2 (Pseudonormal)', color: '#f59e0b' }
      if (text.includes('impaired relaxation') || text.includes('mild')) return { grade: 1, label: 'Grade 1 (Impaired Relaxation)', color: '#eab308' }
      if (text.includes('normal')) return { grade: 0, label: 'Normal', color: '#10b981' }
      return { grade: 0, label: 'Normal', color: '#10b981' }
    }
    
    // LA Enlargement
    if (category === 'Left Atrium' || category === 'LA_SIZE') {
      if (text.includes('severe')) return { grade: 3, label: 'Severe', color: '#ef4444' }
      if (text.includes('moderate')) return { grade: 2, label: 'Moderate', color: '#f59e0b' }
      if (text.includes('mild')) return { grade: 1, label: 'Mild', color: '#eab308' }
      return { grade: 0, label: 'Normal', color: '#10b981' }
    }
    
    // LV Function
    if (category === 'Left Ventricular Function' || category === 'LV_FUNCTION') {
      if (text.includes('severely reduced')) return { grade: 3, label: 'Severe', color: '#ef4444' }
      if (text.includes('moderately reduced')) return { grade: 2, label: 'Moderate', color: '#f59e0b' }
      if (text.includes('mildly reduced')) return { grade: 1, label: 'Mild', color: '#eab308' }
      if (text.includes('normal')) return { grade: 0, label: 'Normal', color: '#10b981' }
      return { grade: 0, label: 'Normal', color: '#10b981' }
    }
    
    return { grade: 0, label: 'Normal', color: '#10b981' }
  }

  return (
    <div className="result-card" style={{ marginTop: 24 }}>
      <div className="result-title">📊 Disease Severity Grading</div>
      <div className="severity-grid">
        {Object.entries(interpretations).map(([category, interpretation]) => {
          const severity = determineSeverity(category, null, interpretation)
          return (
            <div key={category} className="severity-item" style={{ borderColor: severity.color }}>
              <div className="severity-header">
                <span className="severity-name">{category}</span>
                <span className="severity-grade" style={{ backgroundColor: severity.color }}>Grade {severity.grade}</span>
              </div>
              <div className="severity-label" style={{ color: severity.color }}>{severity.label}</div>
              <div className="severity-bar">
                <div className="severity-bar-fill" style={{ width: `${(severity.grade / 3) * 100}%`, backgroundColor: severity.color }} />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function RiskStratificationPanel({ risk, loading, error }) {
  if (!risk) return null

  const getRiskColor = (score) => {
    if (score < 25) return '#10b981'
    if (score < 50) return '#eab308'
    if (score < 75) return '#f59e0b'
    return '#ef4444'
  }

  const overall = risk.overall || {}
  const hf = risk.heart_failure || {}
  const mort = risk.mortality || {}

  return (
    <div className="result-card" style={{ marginTop: 24 }}>
      <div className="result-title">⚠️ Risk Stratification</div>
      
      {loading && <div className="mini-loading">Calculating risk scores...</div>}
      {error && <div className="analysis-error">{error}</div>}
      
      {!loading && !error && (
        <>
          <div className="risk-cards">
            {/* Overall CVD Risk */}
            <div className="risk-card">
              <div className="risk-card__header">Cardiovascular Risk</div>
              <div className="risk-card__score" style={{ color: getRiskColor(overall.overall_score || 0) }}>
                {(overall.overall_score || 0).toFixed(1)}/100
              </div>
              <div className="risk-card__category">{overall.category || 'Unknown'}</div>
              <div className="risk-card__bar">
                <span style={{ width: `${Math.min(100, overall.overall_score || 0)}%`, backgroundColor: getRiskColor(overall.overall_score || 0) }} />
              </div>
              {overall.percentile && (
                <div className="risk-card__detail">Percentile: {overall.percentile.toFixed(1)}%</div>
              )}
            </div>

            {/* Heart Failure Risk */}
            <div className="risk-card">
              <div className="risk-card__header">Heart Failure Risk</div>
              <div className="risk-card__score" style={{ color: getRiskColor(hf.risk_score || 0) }}>
                {(hf.risk_score || 0).toFixed(1)}/100
              </div>
              <div className="risk-card__category">{hf.risk_category || 'Unknown'}</div>
              {hf.one_year_risk_percent !== undefined && (
                <div className="risk-card__detail">1-Year Risk: {hf.one_year_risk_percent.toFixed(1)}%</div>
              )}
              {hf.five_year_risk_percent !== undefined && (
                <div className="risk-card__detail">5-Year Risk: {hf.five_year_risk_percent.toFixed(1)}%</div>
              )}
            </div>

            {/* Mortality Risk */}
            <div className="risk-card">
              <div className="risk-card__header">Mortality Risk</div>
              <div className="risk-card__score" style={{ color: getRiskColor(mort.one_year_mortality || 0) }}>
                {(mort.one_year_mortality || 0).toFixed(1)}%
              </div>
              <div className="risk-card__category">{mort.risk_category || 'Unknown'}</div>
              {mort.five_year_mortality !== undefined && (
                <div className="risk-card__detail">5-Year: {mort.five_year_mortality.toFixed(1)}%</div>
              )}
              {mort.ten_year_mortality !== undefined && (
                <div className="risk-card__detail">10-Year: {mort.ten_year_mortality.toFixed(1)}%</div>
              )}
            </div>
          </div>

          {/* Contributing Factors */}
          {overall.contributing_factors && Object.keys(overall.contributing_factors).length > 0 && (
            <div style={{ marginTop: 24 }}>
              <h4>Contributing Risk Factors</h4>
              <div className="factors-grid">
                {Object.entries(overall.contributing_factors).map(([factor, score]) => (
                  <div key={factor} className="factor-item">
                    <span>{factor.replace(/_/g, ' ')}</span>
                    <div className="factor-bar">
                      <span style={{ width: `${Math.min(100, Math.max(0, (score / 100) * 100))}%` }} />
                    </div>
                    <span className="factor-score">{(score || 0).toFixed(1)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recommendations */}
          {overall.recommendations && overall.recommendations.length > 0 && (
            <div style={{ marginTop: 24 }}>
              <h4>Clinical Recommendations</h4>
              <ul className="recommendations-list">
                {overall.recommendations.map((rec, idx) => (
                  <li key={idx}>
                    <span className="rec-icon">→</span>
                    {rec}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}
    </div>
  )
}

function ExplainabilityPanel({ explainability, featureImportance, loading, error }) {
  const [plots, setPlots] = useState({ v2: [] })
  const [selectedCategory, setSelectedCategory] = useState(null)

  // Load SHAP plots on mount
  React.useEffect(() => {
    const loadPlots = async () => {
      try {
        const categories = ['LV_FUNCTION', 'LV_SIZE', 'LV_HYPERTROPHY', 'LA_SIZE', 'DIASTOLIC_FUNCTION']
        const plotTypes = ['feature_importance', 'shap_summary_bar', 'shap_summary_dot']
        
        const v2Plots = categories.flatMap(cat => 
          plotTypes.map(type => ({
            name: `${type}_${cat}`,
            path: `${API_BASE}/outputs/comparison_plots/v2_expanded/${type}_${cat}.png`,
            category: cat,
            type: type
          }))
        )
        
        setPlots({ v2: v2Plots })
        if (categories.length > 0) setSelectedCategory(categories[0])
      } catch (err) {
        console.error('Error loading plots:', err)
      }
    }
    loadPlots()
  }, [])

  // Only show if explainability data exists
  if (!explainability && (!featureImportance || featureImportance.length === 0)) return null

  const filteredPlots = selectedCategory 
    ? plots.v2.filter(p => p.category === selectedCategory)
    : []

  return (
    <div className="result-card" style={{ marginTop: 24 }}>
      <div className="result-title">🧠 Explainability & Feature Importance</div>
      
      {loading && <div className="mini-loading">Computing explainability analysis...</div>}
      {error && <div className="analysis-error">{error}</div>}
      
      {!loading && !error && (
        <div>
          <div className="explain-grid">
          {/* SHAP-like Top Contributors */}
          {explainability?.top_contributors && explainability.top_contributors.length > 0 && (
            <div className="explain-panel">
              <h4>Top Contributing Features (SHAP-based)</h4>
              <div className="contributors-list">
                {explainability.top_contributors.map(([feature, value], idx) => (
                  <div key={feature} className="contributor-item">
                    <div className="contributor-rank">#{idx + 1}</div>
                    <div className="contributor-name">{feature}</div>
                    <div className="contributor-bar">
                      <span 
                        className={value >= 0 ? 'pos' : 'neg'}
                        style={{ width: `${Math.min(100, Math.abs(value) * 100)}%` }}
                      />
                    </div>
                    <div className="contributor-value">{value >= 0 ? '+' : ''}{Number(value).toFixed(4)}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Prediction Info */}
          {explainability?.prediction && (
            <div className="explain-panel" style={{ gridColumn: '1 / -1' }}>
              <h4>Model Prediction</h4>
              <div style={{ padding: 12, backgroundColor: '#f3f4f6', borderRadius: 6, marginTop: 8 }}>
                <div style={{ fontSize: '1.1rem', fontWeight: 600, color: '#333' }}>
                  Predicted Category: <span style={{ color: '#8b5cf6' }}>{explainability.prediction}</span>
                </div>
                {explainability.feature_values && (
                  <div style={{ fontSize: '0.85rem', color: '#666', marginTop: 8 }}>
                    <strong>Input Values:</strong>
                    <div style={{ marginTop: 6, maxHeight: 120, overflowY: 'auto', fontSize: '0.8rem' }}>
                      {Object.entries(explainability.feature_values).map(([k, v]) => (
                        <div key={k} style={{ padding: '2px 0', display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #e5e7eb' }}>
                          <span>{k}:</span>
                          <span style={{ fontFamily: 'monospace', color: '#555' }}>{typeof v === 'number' ? v.toFixed(2) : v}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
          </div>

          {/* SHAP Plots Section */}
          {plots.v2.length > 0 && (
            <div style={{ marginTop: 24, paddingTop: 24, borderTop: '2px solid #e5e7eb' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
                <h4 style={{ margin: 0 }}>📊 SHAP Model Explainability Plots</h4>
                <select 
                  value={selectedCategory || ''} 
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  style={{ padding: '6px 12px', borderRadius: 4, border: '1px solid #d1d5db', fontSize: '0.9rem' }}
                >
                  {['LV_FUNCTION', 'LV_SIZE', 'LV_HYPERTROPHY', 'LA_SIZE', 'DIASTOLIC_FUNCTION'].map(cat => (
                    <option key={cat} value={cat}>{cat}</option>
                  ))}
                </select>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: 16 }}>
                {filteredPlots.map((plot, idx) => (
                  <div 
                    key={idx}
                    style={{ 
                      border: '1px solid #e5e7eb',
                      borderRadius: 8,
                      overflow: 'hidden',
                      background: 'white'
                    }}
                  >
                    <div style={{ 
                      padding: 10, 
                      background: '#f9fafb',
                      borderBottom: '1px solid #e5e7eb',
                      fontSize: '0.85rem',
                      fontWeight: 600,
                      color: '#666'
                    }}>
                      {plot.type === 'feature_importance' && '🎯 Feature Importance'}
                      {plot.type === 'shap_summary_bar' && '📊 SHAP Summary (Bar)'}
                      {plot.type === 'shap_summary_dot' && '🌐 SHAP Summary (Beeswarm)'}
                    </div>
                    <div style={{ padding: 8, background: '#fafafa', maxHeight: 450, overflow: 'auto' }}>
                      <img 
                        src={plot.path} 
                        alt={plot.name}
                        style={{ width: '100%', height: 'auto', display: 'block', borderRadius: 4 }}
                        onError={(e) => {
                          e.target.style.display = 'none'
                          e.target.parentElement.innerHTML = `<div style="padding: 40px 20px; text-align: center; color: #999; font-size: 0.85rem;">Plot not found</div>`
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: 12, fontSize: '0.8rem', color: '#999' }}>
                💡 <strong>About SHAP Plots:</strong> Feature Importance shows top model features. SHAP Summary (Bar) displays mean feature impacts. SHAP Summary (Beeswarm) shows individual sample contributions color-coded by feature values.
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function SensitivityPanel({ sensitivity, loading, error }) {
  if (!sensitivity || Object.keys(sensitivity).length === 0) return null

  return (
    <div className="result-card" style={{ marginTop: 24 }}>
      <div className="result-title">📈 Sensitivity Analysis</div>
      <p style={{ color: '#666', marginBottom: 16, fontSize: '0.9rem' }}>Shows how predictions change when input parameters vary ±15%</p>
      
      {loading && <div className="mini-loading">Running sensitivity analysis...</div>}
      {error && <div className="analysis-error">{error}</div>}
      
      {!loading && !error && (
        <div className="sensitivity-grid">
          {Object.entries(sensitivity).map(([feature, rows]) => {
            const labels = rows.map(r => (r.variation_percent ?? 0).toFixed(0))
            const data = rows.map(r => r.prediction ?? 0)
            return (
              <div key={feature} className="sensitivity-chart">
                <LineChart title={feature} labels={labels} data={data} color="#06b6d4" yLabel="Prediction Robustness" />
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

function MetricsTab() {
  const [info, setInfo] = useState('Fetching model performance metrics...')
  const [error, setError] = useState('')
  const [metricsV1, setMetricsV1] = useState(null)
  const [metricsV2, setMetricsV2] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [version, setVersion] = useState('v2')
  const [cat, setCat] = useState('OVERALL')
  const [algo, setAlgo] = useState('ALL')

  const accRef = useRef(); const f1Ref = useRef(); const prRef = useRef(); const timeRef = useRef()

  useEffect(() => {
    (async () => {
      try {
        const [v1Res, v2Res, compRes] = await Promise.allSettled([
          fetch(`${API_BASE}/api/model-metrics`),
          fetch(`${API_BASE}/api/model-metrics?version=v2`),
          fetch(`${API_BASE}/api/model-comparison`)
        ])

        let v1Json = null
        let v2Json = null
        let compJson = null
        
        if (v1Res.status === 'fulfilled') {
          const res = v1Res.value
          const json = await res.json()
          if (res.ok) v1Json = json
        }
        if (v2Res.status === 'fulfilled') {
          const res = v2Res.value
          const json = await res.json()
          if (res.ok) v2Json = json
        }
        if (compRes.status === 'fulfilled') {
          const res = compRes.value
          const json = await res.json()
          if (res.ok) compJson = json
        }

        if (!v1Json && !v2Json) throw new Error('Failed to load metrics for v1 and v2')

        if (v1Json) setMetricsV1(v1Json)
        if (v2Json) setMetricsV2(v2Json)
        if (compJson) setComparison(compJson)

        const parts = []
        if (v1Json?.generated_at) parts.push(`v1: ${new Date(v1Json.generated_at * 1000).toLocaleString()}`)
        if (v2Json?.generated_at) parts.push(`v2: ${new Date(v2Json.generated_at * 1000).toLocaleString()}`)
        setInfo(parts.length ? `Metrics generated at ${parts.join(' • ')}` : '')
        setError('')
      } catch (e) {
        setError(`❌ ${e.message}`)
        setInfo('')
      }
    })()
  }, [])

  const metrics = version === 'v2' ? metricsV2 : metricsV1
  const categories = useMemo(() => Object.keys(metrics?.categories || {}), [metrics])
  const algorithms = useMemo(() => {
    if (!metrics) return []
    if (cat === 'OVERALL') {
      const set = new Set()
      for (const c in metrics.categories) { Object.keys(metrics.categories[c].algorithms || {}).forEach(a => set.add(a)) }
      return Array.from(set)
    }
    return Object.keys(metrics.categories?.[cat]?.algorithms || {})
  }, [metrics, cat])

  const computeOverall = (m) => {
    if (!m) return null
    let totalAcc=0, totalPrec=0, totalRec=0, totalF1=0, totalTrain=0, totalPred=0, count=0
    const catEntries = Object.entries(m.categories || {})
    for (const [, cdata] of catEntries) {
      const algoNames = Object.keys(cdata.algorithms || {})
      for (const a of algoNames) {
        const d = cdata.algorithms[a]
        const acc = +d.accuracy || 0, prec=+d.precision||0, rec=+d.recall||0, f1=+d.f1_score||0, tr=+d.train_time||0, prd=+d.predict_time||0
        totalAcc+=acc; totalPrec+=prec; totalRec+=rec; totalF1+=f1; totalTrain+=tr; totalPred+=prd; count++
      }
    }
    return { acc: count? totalAcc/count:0, prec: count? totalPrec/count:0, rec: count? totalRec/count:0, f1: count? totalF1/count:0, tr: count? totalTrain/count:0, prd: count? totalPred/count:0 }
  }

  const overallV1 = useMemo(() => computeOverall(metricsV1), [metricsV1])
  const overallV2 = useMemo(() => computeOverall(metricsV2), [metricsV2])

  const { overall, series } = useMemo(() => {
    if (!metrics) return { overall: null, series: null }
    let totalAcc=0, totalPrec=0, totalRec=0, totalF1=0, totalTrain=0, totalPred=0, count=0
    const catEntries = Object.entries(metrics.categories)
    const filteredCatEntries = cat === 'OVERALL' ? catEntries : catEntries.filter(([c]) => c === cat)
    const accSeries = []; const f1Series = []; const prSeries = []; const timeSeries = []
    for (const [c, cdata] of filteredCatEntries) {
      const algoNames = Object.keys(cdata.algorithms)
      const wanted = algo === 'ALL' || algo === 'OVERALL' ? algoNames : (algoNames.includes(algo) ? [algo] : algoNames)
      for (const a of wanted) {
        const d = cdata.algorithms[a]
        const acc = +d.accuracy || 0, prec=+d.precision||0, rec=+d.recall||0, f1=+d.f1_score||0, tr=+d.train_time||0, prd=+d.predict_time||0
        totalAcc+=acc; totalPrec+=prec; totalRec+=rec; totalF1+=f1; totalTrain+=tr; totalPred+=prd; count++
      }
      accSeries.push({ label: c, data: wanted.map(a => +cdata.algorithms[a]?.accuracy || 0), labels: wanted })
      f1Series.push({ label: c, data: wanted.map(a => +cdata.algorithms[a]?.f1_score || 0), labels: wanted })
      prSeries.push({ label: c, precision: wanted.map(a => +cdata.algorithms[a]?.precision || 0), recall: wanted.map(a => +cdata.algorithms[a]?.recall || 0), labels: wanted })
      timeSeries.push({ label: c, train: wanted.map(a => (+cdata.algorithms[a]?.train_time || 0) * 1000), predict: wanted.map(a => (+cdata.algorithms[a]?.predict_time || 0) * 1000), labels: wanted })
    }
    const avg = { acc: count? totalAcc/count:0, prec: count? totalPrec/count:0, rec: count? totalRec/count:0, f1: count? totalF1/count:0, tr: count? totalTrain/count:0, prd: count? totalPred/count:0 }
    return { overall: avg, series: { accSeries, f1Series, prSeries, timeSeries } }
  }, [metrics, cat, algo])

  useEffect(() => {
    if (!series || cat === 'OVERALL') return
    const cur = metrics.categories?.[cat]
    if (!cur) return

    const labels = (algo === 'ALL' || algo === 'OVERALL') ? Object.keys(cur.algorithms) : [algo]
    const accs = labels.map(a => +cur.algorithms[a]?.accuracy || 0)
    const f1s = labels.map(a => +cur.algorithms[a]?.f1_score || 0)
    const precisions = labels.map(a => +cur.algorithms[a]?.precision || 0)
    const recalls = labels.map(a => +cur.algorithms[a]?.recall || 0)
    const trainTimes = labels.map(a => (+cur.algorithms[a]?.train_time || 0) * 1000)
    const predictTimes = labels.map(a => (+cur.algorithms[a]?.predict_time || 0) * 1000)

    const conf = (ref, cfg) => {
      if (!ref.current) return; const ctx = ref.current.getContext('2d'); if (ref.current.__chart) { ref.current.__chart.destroy() } ; ref.current.__chart = new Chart(ctx, cfg)
    }

    const commonOpts = {
      responsive: true,
      maintainAspectRatio: true,
      aspectRatio: 1.8,
      plugins: { legend: { position: 'bottom' } },
      scales: { x: { ticks: { maxRotation: 0, autoSkip: true } } }
    }

    conf(accRef, { type: 'bar', data: { labels, datasets: [{ label: `${cat} Accuracy`, data: accs, backgroundColor: '#667eea', borderRadius: 6, maxBarThickness: 42, barPercentage: 0.7, categoryPercentage: 0.6 }] }, options: { ...commonOpts, scales: { ...commonOpts.scales, y: { suggestedMin: 0, suggestedMax: 1 } } } })
    conf(f1Ref, { type: 'bar', data: { labels, datasets: [{ label: `${cat} F1-Score`, data: f1s, backgroundColor: '#764ba2', borderRadius: 6, maxBarThickness: 42, barPercentage: 0.7, categoryPercentage: 0.6 }] }, options: { ...commonOpts, scales: { ...commonOpts.scales, y: { suggestedMin: 0, suggestedMax: 1 } } } })
    conf(prRef, { type: 'bar', data: { labels, datasets: [{ label: 'Precision', data: precisions, backgroundColor: '#4caf50', borderRadius: 6, maxBarThickness: 42, barPercentage: 0.7, categoryPercentage: 0.6 }, { label: 'Recall', data: recalls, backgroundColor: '#ff9800', borderRadius: 6, maxBarThickness: 42, barPercentage: 0.7, categoryPercentage: 0.6 }] }, options: { ...commonOpts, scales: { ...commonOpts.scales, y: { suggestedMin: 0, suggestedMax: 1 } } } })
    conf(timeRef, { type: 'bar', data: { labels, datasets: [{ label: 'Training Time (ms)', data: trainTimes, backgroundColor: '#e91e63', borderRadius: 6, maxBarThickness: 42, barPercentage: 0.7, categoryPercentage: 0.6 }, { label: 'Inference Time (ms)', data: predictTimes, backgroundColor: '#03a9f4', borderRadius: 6, maxBarThickness: 42, barPercentage: 0.7, categoryPercentage: 0.6 }] }, options: { ...commonOpts, scales: { ...commonOpts.scales, y: { title: { display: true, text: 'Time (ms)' }, beginAtZero: true } } } })
  }, [series, cat, algo, metrics])

  if (error) return (
    <div className="tab-content active">
      <Alert type="error" message={error} hidden={false} />
    </div>
  )

  return (
    <div className="tab-content active">
      {info && <Alert type="info" message={info} hidden={false} />}

      {comparison && (
        <div className="card" style={{ marginBottom: 20, background: '#f8f9fa', border: '1px solid #e0e0e0' }}>
          <div className="result-title" style={{ color: '#2c3e50', marginBottom: 20, borderBottom: '2px solid #3498db', paddingBottom: 10 }}>
            Model Performance Comparison
          </div>
          
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 20, marginBottom: 20 }}>
            {/* V1 Stats */}
            <div style={{ background: 'white', padding: 20, borderRadius: 6, border: '2px solid #95a5a6', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
              <div style={{ fontSize: '0.9rem', color: '#7f8c8d', marginBottom: 15, fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Version 1 (Original)
              </div>
              <div style={{ display: 'grid', gap: 12, fontSize: '0.9rem', color: '#2c3e50' }}>
                <div>
                  <div style={{ color: '#7f8c8d', fontSize: '0.85rem', marginBottom: 4 }}>Test Accuracy</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#34495e' }}>
                    {comparison.v1?.avg_test_accuracy ? (comparison.v1.avg_test_accuracy * 100).toFixed(1) + '%' : 'N/A'}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#7f8c8d', fontSize: '0.85rem', marginBottom: 4 }}>F1-Score</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: '600', color: '#34495e' }}>
                    {comparison.v1?.avg_test_f1 ? (comparison.v1.avg_test_f1).toFixed(3) : 'N/A'}
                  </div>
                </div>
                <div style={{ borderTop: '1px solid #ecf0f1', paddingTop: 10 }}>
                  <div style={{ color: '#7f8c8d', fontSize: '0.85rem', marginBottom: 4 }}>Training Samples</div>
                  <div style={{ fontSize: '1rem', fontWeight: '600', color: '#34495e' }}>{comparison.v1?.total_train_samples || 0}</div>
                </div>
                <div>
                  <div style={{ color: '#7f8c8d', fontSize: '0.85rem', marginBottom: 4 }}>Test Samples</div>
                  <div style={{ fontSize: '1rem', fontWeight: '600', color: '#34495e' }}>{comparison.v1?.total_test_samples || 0}</div>
                </div>
                <div style={{ borderTop: '1px solid #ecf0f1', paddingTop: 10 }}>
                  <div style={{ color: '#7f8c8d', fontSize: '0.85rem', marginBottom: 4 }}>Generalization Gap</div>
                  <div style={{ fontSize: '1rem', fontWeight: '600', color: comparison.v1?.avg_generalization_gap > 0.05 ? '#e67e22' : '#27ae60' }}>
                    {comparison.v1?.avg_generalization_gap !== null ? (comparison.v1.avg_generalization_gap).toFixed(3) : 'N/A'}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#95a5a6', marginTop: 3 }}>(train - test accuracy)</div>
                </div>
              </div>
            </div>

            {/* V2 Stats */}
            <div style={{ background: 'white', padding: 20, borderRadius: 6, border: '2px solid #3498db', boxShadow: '0 4px 8px rgba(52,152,219,0.15)' }}>
              <div style={{ fontSize: '0.9rem', color: '#3498db', marginBottom: 15, fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Version 2 (Expanded)
              </div>
              <div style={{ display: 'grid', gap: 12, fontSize: '0.9rem', color: '#2c3e50' }}>
                <div>
                  <div style={{ color: '#7f8c8d', fontSize: '0.85rem', marginBottom: 4 }}>Test Accuracy</div>
                  <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#2980b9' }}>
                    {comparison.v2?.avg_test_accuracy ? (comparison.v2.avg_test_accuracy * 100).toFixed(1) + '%' : 'N/A'}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#7f8c8d', fontSize: '0.85rem', marginBottom: 4 }}>F1-Score</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: '600', color: '#2980b9' }}>
                    {comparison.v2?.avg_test_f1 ? (comparison.v2.avg_test_f1).toFixed(3) : 'N/A'}
                  </div>
                </div>
                <div style={{ borderTop: '1px solid #ecf0f1', paddingTop: 10 }}>
                  <div style={{ color: '#7f8c8d', fontSize: '0.85rem', marginBottom: 4 }}>Training Samples</div>
                  <div style={{ fontSize: '1rem', fontWeight: '600', color: '#2980b9' }}>{comparison.v2?.total_train_samples || 0}</div>
                </div>
                <div>
                  <div style={{ color: '#7f8c8d', fontSize: '0.85rem', marginBottom: 4 }}>Test Samples</div>
                  <div style={{ fontSize: '1rem', fontWeight: '600', color: '#2980b9' }}>{comparison.v2?.total_test_samples || 0}</div>
                </div>
                <div style={{ borderTop: '1px solid #ecf0f1', paddingTop: 10 }}>
                  <div style={{ color: '#7f8c8d', fontSize: '0.85rem', marginBottom: 4 }}>Generalization Gap</div>
                  <div style={{ fontSize: '1rem', fontWeight: '600', color: comparison.v2?.avg_generalization_gap > 0.05 ? '#e67e22' : '#27ae60' }}>
                    {comparison.v2?.avg_generalization_gap !== null ? (comparison.v2.avg_generalization_gap).toFixed(3) : 'N/A'}
                  </div>
                  <div style={{ fontSize: '0.75rem', color: '#95a5a6', marginTop: 3 }}>(train - test accuracy)</div>
                </div>
              </div>
            </div>

            {/* Improvement Metrics */}
            <div style={{ background: 'white', padding: 20, borderRadius: 6, border: '2px solid #27ae60', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
              <div style={{ fontSize: '0.9rem', color: '#27ae60', marginBottom: 15, fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                Improvements
              </div>
              <div style={{ display: 'grid', gap: 12, fontSize: '0.9rem', color: '#2c3e50' }}>
                <div>
                  <div style={{ color: '#7f8c8d', fontSize: '0.85rem', marginBottom: 4 }}>Winner</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: '700', color: '#27ae60', textTransform: 'uppercase' }}>
                    Version {comparison.comparison?.winner === 'v2' ? '2' : '1'}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#7f8c8d', fontSize: '0.85rem', marginBottom: 4 }}>Accuracy Improvement</div>
                  <div style={{ fontSize: '1.3rem', fontWeight: '700', color: '#27ae60' }}>
                    {comparison.comparison?.accuracy_improvement !== undefined 
                      ? (comparison.comparison.accuracy_improvement >= 0 ? '+' : '') + (comparison.comparison.accuracy_improvement * 100).toFixed(2) + '%' 
                      : 'N/A'}
                  </div>
                </div>
                <div>
                  <div style={{ color: '#7f8c8d', fontSize: '0.85rem', marginBottom: 4 }}>Relative Improvement</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: '600', color: '#27ae60' }}>
                    {comparison.comparison?.relative_improvement_percent !== undefined 
                      ? (comparison.comparison.relative_improvement_percent >= 0 ? '+' : '') + (comparison.comparison.relative_improvement_percent).toFixed(1) + '%' 
                      : 'N/A'}
                  </div>
                </div>
                <div style={{ borderTop: '1px solid #ecf0f1', paddingTop: 10 }}>
                  <div style={{ color: '#7f8c8d', fontSize: '0.85rem', marginBottom: 4 }}>Additional Training Data</div>
                  <div style={{ fontSize: '1rem', fontWeight: '600', color: '#27ae60' }}>
                    +{(comparison.v2?.total_train_samples || 0) - (comparison.v1?.total_train_samples || 0)} samples
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div style={{ background: '#e8f4f8', padding: 15, borderRadius: 6, fontSize: '0.88rem', color: '#2c3e50', border: '1px solid #d0e8f0' }}>
            <strong style={{ color: '#2980b9' }}>Analysis:</strong> Version {comparison.comparison?.winner === 'v2' ? '2' : '1'} demonstrates superior performance. 
            The generalization gap (difference between training and test accuracy) indicates overfitting risk. Values near zero suggest good generalization, 
            while values exceeding 0.05 may indicate the model is memorizing training data rather than learning patterns.
          </div>
        </div>
      )}

      <div className="card" style={{ marginBottom: 20, background: 'linear-gradient(135deg, #2a5298 0%, #1e3c72 100%)', color: 'white' }}>
        <div className="result-title" style={{ color: 'white', marginBottom: 15 }}>🧪 Algorithm Performance Comparison (v1 vs v2)</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 15 }}>
          <MetricTile label="v1 Avg Accuracy" value={`${((overallV1?.acc||0) * 100).toFixed(1)}%`} />
          <MetricTile label="v2 Avg Accuracy" value={`${((overallV2?.acc||0) * 100).toFixed(1)}%`} />
          <MetricTile label="Δ Accuracy (v2 - v1)" value={`${(((overallV2?.acc||0) - (overallV1?.acc||0)) * 100).toFixed(1)}%`} />
          <MetricTile label="Δ F1-Score (v2 - v1)" value={`${(((overallV2?.f1||0) - (overallV1?.f1||0))).toFixed(3)}`} />
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
        <div className="result-title" style={{ color: 'white', marginBottom: 15 }}>📊 Overall Model Performance</div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 15 }}>
          <MetricTile label="📈 Avg Accuracy" value={`${((overall?.acc||0) * 100).toFixed(1)}%`} />
          <MetricTile label="🎯 Avg Precision" value={`${((overall?.prec||0) * 100).toFixed(1)}%`} />
          <MetricTile label="🔍 Avg Recall" value={`${((overall?.rec||0) * 100).toFixed(1)}%`} />
          <MetricTile label="⚡ Avg F1-Score" value={`${(overall?.f1||0).toFixed(3)}`} />
        </div>
        <div id="overallStats" style={{ marginTop: 15, paddingTop: 15, borderTop: '1px solid rgba(255,255,255,0.2)', fontSize: '0.9rem', opacity: 0.9 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 10 }}>
            <div><strong>🏷️ Filter:</strong> {cat === 'OVERALL' ? (algo === 'ALL' ? 'All Categories & Algorithms' : `All Categories • ${algo}`) : `${cat}${algo==='ALL'?'':' • ' + algo}`}</div>
            <div><strong>⏱ Avg Training:</strong> {((overall?.tr||0)*1000).toFixed(1)} ms</div>
            <div><strong>⚡ Avg Inference:</strong> {((overall?.prd||0)*1000).toFixed(2)} ms</div>
          </div>
        </div>
      </div>

      <div className="grid" style={{ alignItems: 'end', marginBottom: 15 }}>
        <div className="form-group">
          <label htmlFor="metricsVersion">Version</label>
          <select id="metricsVersion" value={version} onChange={e => setVersion(e.target.value)}>
            <option value="v1">v1 (Original)</option>
            <option value="v2">v2 (Expanded)</option>
          </select>
        </div>
        <div className="form-group">
          <label htmlFor="metricsCategory">Category</label>
          <select id="metricsCategory" value={cat} onChange={e => { setCat(e.target.value); setAlgo('ALL') }}>
            <option value="OVERALL">📊 Overall Results</option>
            {categories.map(c => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <div className="form-group">
          <label htmlFor="metricsAlgorithm">Algorithm</label>
          <select id="metricsAlgorithm" value={algo} onChange={e => setAlgo(e.target.value)}>
            <option value="ALL">All algorithms</option>
            {algorithms.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
        </div>
      </div>

      {cat !== 'OVERALL' && (
        <>
          <div className="grid">
            <div className="result-card"><div className="result-title">📈 Accuracy by Algorithm</div><canvas ref={accRef} className="chart-canvas" /></div>
            <div className="result-card"><div className="result-title">🎯 F1-Score by Algorithm</div><canvas ref={f1Ref} className="chart-canvas" /></div>
          </div>
          <div className="grid" style={{ marginTop: 20 }}>
            <div className="result-card"><div className="result-title">🎯 Precision & Recall</div><canvas ref={prRef} className="chart-canvas" /></div>
            <div className="result-card"><div className="result-title">⏱ Training & Inference Time (ms)</div><canvas ref={timeRef} className="chart-canvas" /></div>
          </div>
        </>
      )}

      {cat !== 'OVERALL' && metrics?.categories?.[cat]?.algorithms && (
        <ConfusionMatrix category={cat} algo={algo} metrics={metrics} />
      )}
    </div>
  )
}

function MetricTile({ label, value }) {
  return (
    <div style={{ padding: 15, background: 'rgba(255,255,255,0.1)', borderRadius: 8, backdropFilter: 'blur(10px)' }}>
      <div style={{ fontSize: '0.9rem', opacity: 0.9, marginBottom: 5 }}>{label}</div>
      <div style={{ fontSize: '1.8rem', fontWeight: 700 }}>{value}</div>
    </div>
  )
}

function ConfusionMatrix({ category, algo, metrics }) {
  const cmDiv = useRef()
  useEffect(() => {
    const el = cmDiv.current
    if (!el) return
    const catData = metrics.categories?.[category]
    const algoNames = Object.keys(catData.algorithms)
    const chosen = (algo === 'ALL' || algo === 'OVERALL') ? (catData.best_algorithm || algoNames[0]) : algo
    const cm = catData.algorithms[chosen]?.confusion_matrix
    if (!cm || !cm.labels || !cm.matrix || !cm.matrix.length) { el.innerHTML = '<div class="alert alert-info">No confusion matrix available for this selection.</div>'; return }

    const labels = cm.labels
    const matrix = cm.matrix
    let maxVal = 0, totalCorrect = 0, totalSamples = 0
    matrix.forEach((row, i) => row.forEach((v, j) => { if (v > maxVal) maxVal = v; totalSamples += v; if (i === j) totalCorrect += v }))
    const accuracy = totalSamples > 0 ? (totalCorrect / totalSamples * 100).toFixed(1) : 0

    let html = `<div style="overflow-x: auto; margin-bottom: 15px;">
      <p style="font-size: 0.95rem; color: #666; margin-bottom: 10px;">
        <strong>Accuracy:</strong> ${accuracy}% • <strong>Samples:</strong> ${totalSamples} • <strong>Correct:</strong> ${totalCorrect}
      </p>
      <table style="border-collapse: collapse; font-family: 'Courier New', monospace; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">`;
    html += '<tr>';
    html += '<th style="padding:12px 15px; border:1px solid #ddd; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:white; font-weight:600; text-align:center; min-width:80px;">Actual \\ Predicted</th>';
    labels.forEach(l => { html += `<th style=\"padding:12px 15px; border:1px solid #ddd; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:white; font-weight:600; text-align:center; min-width:80px;\">${l}</th>` });
    html += '</tr>';
    for (let i = 0; i < labels.length; i++) {
      html += '<tr>'
      html += `<th style=\"padding:12px 15px; border:1px solid #ddd; background:#f5f5f7; font-weight:600; text-align:center;\">${labels[i]}</th>`
      for (let j = 0; j < labels.length; j++) {
        const val = matrix[i][j] || 0
        const intensity = maxVal ? (val / maxVal) : 0
        const isCorrect = i === j
        let bg, textColor, fontWeight
        if (isCorrect) { const g = 0.2 + 0.7 * intensity; bg = `rgba(76, 175, 80, ${g})`; textColor = intensity > 0.5 ? 'white' : '#1a1a1a'; fontWeight = '700' }
        else { const r = 0.2 + 0.6 * intensity; bg = `rgba(244, 67, 54, ${r})`; textColor = intensity > 0.4 ? 'white' : '#1a1a1a'; fontWeight = intensity > 0.3 ? '600' : '500' }
        html += `<td style=\"padding:12px 15px; border:1px solid #ddd; text-align:center; background:${bg}; color:${textColor}; font-weight:${fontWeight}; font-size:0.95rem;\">${val}</td>`
      }
      html += '</tr>'
    }
    html += '</table></div>'
    html += `<div style=\"display:flex; gap:20px; margin-top:15px; font-size:0.9rem;\">
      <div style=\"display:flex; align-items:center; gap:8px;\"><div style=\"width:20px; height:20px; background:rgba(76, 175, 80, 0.7); border-radius:3px;\"></div><span>True Positives (Correct)</span></div>
      <div style=\"display:flex; align-items:center; gap:8px;\"><div style=\"width:20px; height:20px; background:rgba(244, 67, 54, 0.7); border-radius:3px;\"></div><span>False Positives/Negatives (Errors)</span></div>
    </div>`
    el.innerHTML = html
  }, [category, algo, metrics])
  return (
    <div className="card" style={{ marginTop: 20 }}>
      <div className="result-title">🧩 Confusion Matrix</div>
      <div ref={cmDiv} style={{ overflowX: 'auto', marginTop: 10 }} />
    </div>
  )
}

function ResultsCard({ results }) {
  if (!results) return null
  const patient = results.patient || {}
  const measurements = results.measurements || {}
  const interpretations = results.interpretations || {}
  const severity_grading = results.severity_grading || {}
  const [q, setQ] = useState('')
  const [sort, setSort] = useState('alpha') // alpha | value
  const [analysisCategory, setAnalysisCategory] = useState('')
  const [availableCategories, setAvailableCategories] = useState([])
  const [explainability, setExplainability] = useState(null)
  const [featureImportance, setFeatureImportance] = useState([])
  const [sensitivity, setSensitivity] = useState(null)
  const [risk, setRisk] = useState(null)
  const [analysisLoading, setAnalysisLoading] = useState({ explain: false, importance: false, sensitivity: false, risk: false })
  const [analysisError, setAnalysisError] = useState({ explain: '', importance: '', sensitivity: '', risk: '' })

  const UNITS = {
    EF: '%', FS: '%',
    LVID_D: 'cm', LVID_S: 'cm', LVPW_D: 'cm', LVPW_S: 'cm', IVS_D: 'cm', IVS_S: 'cm',
    LA_DIMENSION: 'cm', AORTIC_ROOT: 'cm', LA_AO: '',
    EDV: 'ml', ESV: 'ml',
    MV_E: 'cm/s', MV_A: 'cm/s', MV_E_A: '', AI_MAX_VEL: 'cm/s', MAX_PG_AI: 'mmHg',
  }

  function prettyLabel(key) {
    // Prefer known prettifications
    const map = { LVID_D: 'LVIDd', LVID_S: 'LVIDs', IVS_D: 'IVSd', IVS_S: 'IVSs', LVPW_D: 'LVPWd', LVPW_S: 'LVPWs', LA_DIMENSION: 'LA Dimension', AORTIC_ROOT: 'Aortic Root', LA_AO: 'LA/Ao', AI_MAX_VEL: 'AI Max Vel', MAX_PG_AI: 'Max PG (AI)', MV_E_A: 'MV E/A' }
    if (map[key]) return map[key]
    return key.replace(/_/g, ' ')
  }

  function classifySeverity(text) {
    const t = (text || '').toLowerCase()
    if (t.includes('severe') || t.includes('restrictive')) return 'severe'
    if (t.includes('moderate')) return 'moderate'
    if (t.includes('mild') || t.includes('borderline') || t.includes('impaired relaxation')) return 'mild'
    if (t.includes('normal')) return 'normal'
    if (t.includes('dysfunction') || t.includes('dilat') || t.includes('enlarg') || t.includes('abnormal')) return 'moderate'
    return 'normal'
  }

  function mapSeverityToClass(sev) {
    switch (sev) {
      case 'severe':
        return 'critical'
      case 'moderate':
      case 'mild':
        return 'warning'
      default:
        return 'normal'
    }
  }

  function copyJSON() {
    navigator.clipboard.writeText(JSON.stringify(results, null, 2))
      .then(() => alert('JSON copied to clipboard!'))
  }

  async function fetchExplainability(categoryOverride) {
    setAnalysisLoading(s => ({ ...s, explain: true }))
    setAnalysisError(s => ({ ...s, explain: '' }))
    try {
      const res = await fetch(`${API_BASE}/api/explainability`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ measurements, patient, category: categoryOverride || analysisCategory || undefined })
      })
      const json = await res.json()
      if (!res.ok) throw new Error(json.error || 'Explainability failed')
      
      const report = json.report || null
      const importance = report?.feature_importance || []
      
      setExplainability(report)
      setFeatureImportance(importance)
      setAvailableCategories(json.available_categories || [])
      if (!analysisCategory && json.category) setAnalysisCategory(json.category)
    } catch (e) {
      setAnalysisError(s => ({ ...s, explain: `❌ ${e.message}` }))
      setExplainability(null)
      setFeatureImportance([])
    } finally {
      setAnalysisLoading(s => ({ ...s, explain: false }))
    }
  }

  async function fetchFeatureImportance(categoryOverride) {
    setAnalysisLoading(s => ({ ...s, importance: true }))
    setAnalysisError(s => ({ ...s, importance: '' }))
    try {
      const res = await fetch(`${API_BASE}/api/feature-importance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category: categoryOverride || analysisCategory || undefined, top_n: 10 })
      })
      const json = await res.json()
      if (!res.ok) throw new Error(json.error || 'Feature importance failed')
      setFeatureImportance(json.feature_importance || [])
      setAvailableCategories(json.available_categories || [])
      if (!analysisCategory && json.category) setAnalysisCategory(json.category)
    } catch (e) {
      setAnalysisError(s => ({ ...s, importance: `❌ ${e.message}` }))
      setFeatureImportance([])
    } finally {
      setAnalysisLoading(s => ({ ...s, importance: false }))
    }
  }

  async function fetchSensitivity(categoryOverride) {
    setAnalysisLoading(s => ({ ...s, sensitivity: true }))
    setAnalysisError(s => ({ ...s, sensitivity: '' }))
    try {
      const res = await fetch(`${API_BASE}/api/sensitivity-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ measurements, patient, category: categoryOverride || analysisCategory || undefined, variation_range: [-0.15, 0.15], n_steps: 12, max_features: 8 })
      })
      const json = await res.json()
      if (!res.ok) throw new Error(json.error || 'Sensitivity analysis failed')
      setSensitivity(json.results || null)
      setAvailableCategories(json.available_categories || [])
      if (!analysisCategory && json.category) setAnalysisCategory(json.category)
    } catch (e) {
      setAnalysisError(s => ({ ...s, sensitivity: `❌ ${e.message}` }))
      setSensitivity(null)
    } finally {
      setAnalysisLoading(s => ({ ...s, sensitivity: false }))
    }
  }

  async function fetchRisk() {
    setAnalysisLoading(s => ({ ...s, risk: true }))
    setAnalysisError(s => ({ ...s, risk: '' }))
    try {
      const res = await fetch(`${API_BASE}/api/risk-stratification`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ measurements, patient })
      })
      const json = await res.json()
      if (!res.ok) throw new Error(json.error || 'Risk stratification failed')
      setRisk(json)
    } catch (e) {
      setAnalysisError(s => ({ ...s, risk: `❌ ${e.message}` }))
      setRisk(null)
    } finally {
      setAnalysisLoading(s => ({ ...s, risk: false }))
    }
  }

  useEffect(() => {
    setExplainability(null)
    setFeatureImportance([])
    setSensitivity(null)
    setRisk(null)
    setAnalysisCategory('')
    setAvailableCategories([])
    if (!results) return
    // Don't auto-fetch - user clicks to load
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [results])

  const measEntries = useMemo(() => {
    const entries = Object.entries(measurements)
      .map(([k, v]) => [k, typeof v === 'number' ? v : parseFloat(v)])
    const filtered = q ? entries.filter(([k]) => k.toLowerCase().includes(q.toLowerCase())) : entries
    const sorted = [...filtered].sort((a, b) => {
      if (sort === 'value') return (b[1] || 0) - (a[1] || 0)
      return a[0].localeCompare(b[0])
    })
    return sorted
  }, [measurements, q, sort])

  return (
    <div id="resultsCard" className="card">
      <h2 style={{ marginBottom: 20, color: '#333' }}>📊 Analysis Results</h2>
      <div id="patientInfo">
        <div className="patient-info">
          <h3>👤 Patient Information</h3>
          <div className="patient-detail"><strong>Name:</strong> {patient.name || 'N/A'}</div>
          <div className="patient-detail"><strong>Age:</strong> {patient.age || 'N/A'} years</div>
          <div className="patient-detail"><strong>Sex:</strong> {patient.sex || 'N/A'}</div>
        </div>
      </div>
      <div className="grid">
        <div className="result-card">
          <div className="result-title">📈 Measurements</div>
          <div className="measurements-toolbar">
            <div className="measurements-count">{measEntries.length} {measEntries.length === 1 ? 'item' : 'items'}</div>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
              <div className="measurements-search"><input placeholder="Search measurement…" value={q} onChange={e => setQ(e.target.value)} /></div>
              <select value={sort} onChange={e => setSort(e.target.value)}>
                <option value="alpha">Sort A–Z</option>
                <option value="value">Sort by value</option>
              </select>
            </div>
          </div>
          <div id="measurementsList" className="measurements-grid">
            {measEntries.map(([k, v]) => {
              const unit = UNITS[k] ?? ''
              const display = typeof v === 'number' && !Number.isNaN(v) ? v.toFixed(unit.includes('%') ? 1 : 2) : String(v)
              return (
                <div key={k} className="measurement-item">
                  <span className="measurement-name">{prettyLabel(k)}</span>
                  <span className="measurement-value">{display}{unit ? ` ${unit}` : ''}</span>
                </div>
              )
            })}
          </div>
        </div>
        <div className="result-card">
          <div className="result-title">🩺 Clinical Interpretation</div>
          <div id="interpretationsList">
            {Object.entries(interpretations).map(([category, text]) => {
              const sev = classifySeverity(text)
              const sevClass = mapSeverityToClass(sev)
              const src = (results.sources && results.sources[category]) ? results.sources[category] : ((results.method || '').includes('ML') ? 'ML' : 'Rule')
              const srcClass = src === 'ML' ? 'source-ml' : 'source-rule'
              return (
                <div key={category} className={`interpretation-item ${sevClass}`}>
                  <div className="interpretation-category">{category}
                    <span className={`source-badge ${srcClass}`} title={`Generated by ${src}`}>{src}</span>
                  </div>
                  <div className="interpretation-text">{text}</div>
                </div>
              )
            })}
          </div>
        </div>
      </div>
      <SeverityGrader severity_grading={severity_grading} measurements={measurements} interpretations={interpretations} />

      <div className="result-card" style={{ marginTop: 24 }}>
        <div className="result-title">🔬 Advanced Analysis Tools</div>
        <div className="analysis-buttons">
          <button className="btn btn-secondary" onClick={() => fetchExplainability()}>🧠 Explainability & Feature Importance</button>
          <button className="btn btn-secondary" onClick={() => fetchSensitivity()}>📈 Sensitivity Analysis</button>
          <button className="btn btn-secondary" onClick={() => fetchRisk()}>⚠️ Risk Stratification</button>
        </div>
      </div>

      <ExplainabilityPanel explainability={explainability} featureImportance={featureImportance} loading={analysisLoading.explain || analysisLoading.importance} error={analysisError.explain || analysisError.importance} />

      <SensitivityPanel sensitivity={sensitivity} loading={analysisLoading.sensitivity} error={analysisError.sensitivity} />

      <RiskStratificationPanel risk={risk} loading={analysisLoading.risk} error={analysisError.risk} />
      <div style={{ marginTop: 30 }}>
        <h3 style={{ marginBottom: 15 }}>📋 JSON Response</h3>
        <button className="btn btn-secondary copy-btn" onClick={copyJSON}>📋 Copy JSON</button>
        <div className="json-viewer" id="jsonViewer">{JSON.stringify(results, null, 2)}</div>
      </div>
    </div>
  )
}

export default function App() {
  const [active, setActive] = useState('upload')
  const [results, setResults] = useState(null)

  useEffect(() => {
    ;(async () => {
      try { const res = await fetch(`${API_BASE}/health`); const json = await res.json(); console.log('API Status:', json) } catch (e) { console.warn('API not reachable at', API_BASE) }
    })()
  }, [])

  return (
    <div className="app-container">
      <div className="header">
        <h1>🏥 Medical Report Interpreter</h1>
        <p>AI-Powered Echocardiography Analysis System</p>
      </div>

      <div className="card">
        <Tabs active={active} onChange={setActive} />
        {active === 'upload' && <UploadTab onResults={setResults} />}
        {active === 'manual' && <ManualTab onResults={setResults} />}
        {active === 'batch' && <BatchTab />}
        {active === 'metrics' && <MetricsTab />}
      </div>

      {(active === 'upload' || active === 'manual') && results && (
        <ResultsCard results={results} />
      )}
    </div>
  )
}
