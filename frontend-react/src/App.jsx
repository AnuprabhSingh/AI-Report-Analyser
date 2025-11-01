import React, { useEffect, useMemo, useRef, useState } from 'react'
import { Chart, BarElement, BarController, CategoryScale, LinearScale, Legend, LineElement, LineController, PointElement, Title, Tooltip } from 'chart.js'

Chart.register(BarElement, BarController, CategoryScale, LinearScale, Legend, LineElement, LineController, PointElement, Title, Tooltip)

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:5000'

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

function MetricsTab() {
  const [info, setInfo] = useState('Fetching model performance metrics...')
  const [error, setError] = useState('')
  const [metrics, setMetrics] = useState(null)
  const [cat, setCat] = useState('OVERALL')
  const [algo, setAlgo] = useState('ALL')

  const accRef = useRef(); const f1Ref = useRef(); const prRef = useRef(); const timeRef = useRef()

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/model-metrics`)
        const json = await res.json()
        if (!res.ok) throw new Error(json.error || 'Failed to load metrics')
        setMetrics(json)
        setInfo(`Metrics generated at ${new Date((json.generated_at || Date.now()) * 1000).toLocaleString()}`)
        setError('')
      } catch (e) {
        setError(`❌ ${e.message}`)
        setInfo('')
      }
    })()
  }, [])

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
  const [q, setQ] = useState('')
  const [sort, setSort] = useState('alpha') // alpha | value

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
