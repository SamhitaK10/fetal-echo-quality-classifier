import { motion } from 'framer-motion'
import Heart3D from './Heart3D'

const STATS = [
  { val: '100%',     label: 'Validation Accuracy' },
  { val: '6',        label: 'Quality Classes'      },
  { val: '<1s',      label: 'Inference Time'       },
  { val: 'EigenCAM', label: 'Explainability'       },
]

export default function Hero() {
  return (
    <section className="hero">
      <motion.div
        className="hero-content"
        initial={{ opacity: 0, x: -40 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.8, ease: 'easeOut' }}
      >
        <div className="hero-badge">
          <span className="badge-dot" />
          EfficientNet-B3 · Clinical Grade
        </div>

        <h1 className="hero-title">
          Fetal Cardiac<br />
          <span className="gradient-text">Quality Assessment</span>
        </h1>

        <p className="hero-sub">
          Automated quality classification for first-trimester fetal echocardiography.
          EfficientNet-B3 identifies six quality defects and delivers instant sonographer guidance.
        </p>

        <div className="hero-actions">
          <a href="#demo" className="btn-primary">Open Assessment Tool →</a>
          <a href="#how-it-works" className="btn-secondary">How It Works</a>
        </div>

        <div className="hero-stats">
          {STATS.map(s => (
            <div key={s.label}>
              <span className="stat-val">{s.val}</span>
              <span className="stat-label">{s.label}</span>
            </div>
          ))}
        </div>
      </motion.div>

      <div className="hero-canvas">
        <Heart3D />
      </div>
    </section>
  )
}
