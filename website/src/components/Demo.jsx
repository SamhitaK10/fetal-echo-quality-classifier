import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import html2canvas from 'html2canvas'

const API = import.meta.env.VITE_API_URL ?? 'https://samhitak10-fetal-echo-quality-demo.hf.space'

const CLASS_COLORS = {
  good:         '#22c55e',
  blurry:       '#3b82f6',
  too_dark:     '#ef4444',
  low_contrast: '#f59e0b',
  noisy:        '#a855f7',
  angled:       '#eab308',
}

const CLASS_ICONS = {
  good:         '✓',
  blurry:       '〰',
  too_dark:     '◼',
  low_contrast: '◑',
  noisy:        '⊛',
  angled:       '↗',
}

const CLASS_LABELS = {
  good:         'Diagnostic Quality',
  blurry:       'Motion / Focus Artefact',
  too_dark:     'Insufficient Gain',
  low_contrast: 'Poor Contrast Resolution',
  noisy:        'Acoustic Noise',
  angled:       'Suboptimal Probe Angle',
}

function ProbBars({ probs }) {
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1])
  return (
    <div className="prob-card">
      <div className="prob-card-title">Confidence Distribution</div>
      {sorted.map(([cls, p], i) => (
        <div key={cls} className="bar-row">
          <span className="bar-label">{CLASS_LABELS[cls] ?? cls}</span>
          <div className="bar-track">
            <motion.div
              className="bar-fill"
              initial={{ width: 0 }}
              animate={{ width: `${p * 100}%` }}
              transition={{ duration: 0.75, delay: i * 0.07, ease: 'easeOut' }}
              style={{ background: CLASS_COLORS[cls] || '#3b82f6' }}
            />
          </div>
          <span className="bar-pct">{(p * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

// Lightbox for zoomed image
function Lightbox({ src, onClose }) {
  return (
    <motion.div
      className="lightbox"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
    >
      <motion.img
        src={src}
        className="lightbox-img"
        initial={{ scale: 0.85 }}
        animate={{ scale: 1 }}
        exit={{ scale: 0.85 }}
        onClick={e => e.stopPropagation()}
      />
      <button className="lightbox-close" onClick={onClose}>✕</button>
    </motion.div>
  )
}

export default function Demo() {
  const [tab,      setTab]      = useState('image')
  const [file,     setFile]     = useState(null)
  const [preview,  setPreview]  = useState(null)
  const [dragging, setDragging] = useState(false)
  const [loading,  setLoading]  = useState(false)
  const [result,   setResult]   = useState(null)
  const [videoUrl, setVideoUrl] = useState(null)
  const [error,    setError]    = useState(null)
  const [zoomSrc,  setZoomSrc]  = useState(null)
  const [saving,   setSaving]   = useState(false)
  const inputRef  = useRef()
  const cardRef   = useRef()

  const reset = () => {
    setFile(null); setPreview(null)
    setResult(null); setVideoUrl(null); setError(null)
  }

  const handleFile = useCallback(f => {
    if (!f) return
    setFile(f); setResult(null); setVideoUrl(null); setError(null)
    if (f.type.startsWith('image/')) setPreview(URL.createObjectURL(f))
    else setPreview(null)
  }, [])

  const handleDrop = e => {
    e.preventDefault(); setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  const analyze = async () => {
    if (!file) return
    setLoading(true); setError(null)
    const form = new FormData()
    form.append('file', file)
    try {
      if (tab === 'image') {
        const res  = await fetch(`${API}/predict`, { method: 'POST', body: form })
        if (!res.ok) throw new Error(`Server ${res.status}`)
        const data = await res.json()
        if (data.error) throw new Error(data.error)
        setResult(data)
      } else {
        const res  = await fetch(`${API}/predict-video`, { method: 'POST', body: form })
        if (!res.ok) throw new Error(`Server ${res.status}`)
        setVideoUrl(URL.createObjectURL(await res.blob()))
      }
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const saveAsImage = async () => {
    if (!cardRef.current) return
    setSaving(true)
    try {
      const canvas = await html2canvas(cardRef.current, {
        backgroundColor: '#0d1424',
        scale: 2,
        useCORS: true,
        logging: false,
      })
      const link = document.createElement('a')
      link.download = `echo-assessment-${Date.now()}.png`
      link.href = canvas.toDataURL('image/png')
      link.click()
    } finally {
      setSaving(false)
    }
  }

  const color = result ? (CLASS_COLORS[result.label] ?? '#3b82f6') : '#3b82f6'
  const showUpload = !loading && !result && !videoUrl

  return (
    <section id="demo" className="demo-section">
      <div className="container">
        <div className="demo-header">
          <div className="section-tag">Assessment</div>
          <h2 className="section-title">Quality Analysis</h2>
          <p className="section-sub">
            Submit a fetal echocardiography scan for automated quality classification and clinical guidance.
          </p>
        </div>

        {/* Tabs */}
        <div className="demo-tabs">
          {['image', 'video'].map(t => (
            <button
              key={t}
              className={`tab-btn${tab === t ? ' active' : ''}`}
              onClick={() => { setTab(t); reset() }}
            >
              {t === 'image' ? '🖼 Image' : '🎬 Video Clip'}
            </button>
          ))}
        </div>

        {/* Upload */}
        {showUpload && (
          <>
            <div
              className={`upload-zone${dragging ? ' drag' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => inputRef.current?.click()}
            >
              <input
                ref={inputRef}
                type="file"
                className="file-input"
                accept={tab === 'image' ? 'image/*' : 'video/*'}
                onChange={e => handleFile(e.target.files[0])}
              />
              {file ? (
                <>
                  {preview && <img src={preview} className="file-preview" alt="preview" />}
                  <div className="upload-title">✓ {file.name}</div>
                  <div className="upload-sub">{(file.size / 1024 / 1024).toFixed(2)} MB</div>
                </>
              ) : (
                <>
                  <span className="upload-icon">{tab === 'image' ? '🫁' : '📹'}</span>
                  <div className="upload-title">Drop your {tab === 'image' ? 'scan' : 'clip'} here</div>
                  <div className="upload-sub">or click to browse</div>
                  <div className="upload-hint">
                    {tab === 'image' ? 'PNG · JPG · BMP — max 20 MB' : 'MP4 · AVI · MOV — max 200 MB'}
                  </div>
                </>
              )}
            </div>
            <button className="analyze-btn" onClick={analyze} disabled={!file}>
              ⚡ Analyze
            </button>
          </>
        )}

        {/* Loading */}
        {loading && (
          <div className="loading-wrap">
            <div className="spinner" />
            <p className="loading-txt">
              {tab === 'video' ? 'Processing frames — this may take a minute' : 'Running classification…'}
            </p>
          </div>
        )}

        {error && <div className="error-box">⚠ {error}</div>}

        {/* Result card */}
        <AnimatePresence>
          {result && (
            <motion.div
              className="result-wrap"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.4 }}
            >
              {/* The exportable card */}
              <div ref={cardRef} className="result-card-export">
                <div className="result-header">
                  <div
                    className="result-badge"
                    style={{ background: `${color}18`, color, border: `1px solid ${color}40` }}
                  >
                    <span>{CLASS_ICONS[result.label] ?? '●'}</span>
                    <span>{CLASS_LABELS[result.label] ?? result.label}</span>
                  </div>
                  <span className="result-conf" style={{ color }}>
                    {(result.confidence * 100).toFixed(1)}%
                  </span>
                </div>

                <div className="images-grid">
                  <div className="img-card">
                    <div className="img-card-label">Original Scan</div>
                    <img
                      src={`data:image/jpeg;base64,${result.orig_image}`}
                      alt="original"
                      style={{ cursor: 'zoom-in' }}
                      onClick={() => setZoomSrc(`data:image/jpeg;base64,${result.orig_image}`)}
                    />
                  </div>
                  <div className="img-card">
                    <div className="img-card-label">EigenCAM Activation</div>
                    <img
                      src={`data:image/jpeg;base64,${result.cam_image}`}
                      alt="heatmap"
                      style={{ cursor: 'zoom-in' }}
                      onClick={() => setZoomSrc(`data:image/jpeg;base64,${result.cam_image}`)}
                    />
                  </div>
                </div>

                <ProbBars probs={result.probs} />

                <div className="guidance-card" style={{ borderLeft: `3px solid ${color}` }}>
                  <div className="guidance-icon" style={{ background: `${color}18`, color }}>
                    {result.label === 'good' ? '✓' : '💡'}
                  </div>
                  <div>
                    <div className="guidance-title">Clinical Recommendation</div>
                    <div className="guidance-text">{result.guidance}</div>
                  </div>
                </div>
              </div>

              {/* Action buttons */}
              <div className="result-actions">
                <button
                  className="btn-primary"
                  onClick={saveAsImage}
                  disabled={saving}
                >
                  {saving ? '⟳ Exporting…' : '↓ Save Report'}
                </button>
                <button className="reset-btn" onClick={reset}>
                  ← New Assessment
                </button>
              </div>
            </motion.div>
          )}

          {videoUrl && (
            <motion.div
              className="video-result"
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <video src={videoUrl} controls autoPlay loop />
              <div className="result-actions">
                <a href={videoUrl} download="echo-analysis.mp4" className="btn-primary">
                  ↓ Download Annotated Clip
                </a>
                <button className="reset-btn" onClick={reset}>← New Assessment</button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Zoom lightbox */}
      <AnimatePresence>
        {zoomSrc && <Lightbox src={zoomSrc} onClose={() => setZoomSrc(null)} />}
      </AnimatePresence>
    </section>
  )
}
