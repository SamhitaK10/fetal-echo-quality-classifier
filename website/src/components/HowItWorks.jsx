import { motion } from 'framer-motion'

const STEPS = [
  {
    icon: '📡',
    cls:  'icon-blue',
    title: 'Submit Your Scan',
    desc:  'Upload any first-trimester fetal echo image or short video clip. PNG, JPG, MP4 and AVI are all supported.',
  },
  {
    icon: '🔬',
    cls:  'icon-cyan',
    title: 'Neural Classification',
    desc:  'EfficientNet-B3 classifies the scan into one of six quality categories in under a second, with EigenCAM activation mapping for full transparency.',
  },
  {
    icon: '📋',
    cls:  'icon-purple',
    title: 'Clinical Recommendation',
    desc:  'Receive a tailored recommendation to help improve image quality and support a more accurate cardiac assessment.',
  },
]

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="how-section">
      <div className="container">
        <div className="how-header">
          <div className="section-tag">Workflow</div>
          <h2 className="section-title">How It Works</h2>
          <p className="section-sub">Three steps from acquisition to clinical guidance.</p>
        </div>

        <div className="how-grid">
          {STEPS.map((s, i) => (
            <motion.div
              key={i}
              className="how-card"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: '-60px' }}
              transition={{ duration: 0.5, delay: i * 0.14 }}
            >
              <span className="how-step-num">0{i + 1}</span>
              <div className={`how-icon ${s.cls}`}>{s.icon}</div>
              <h3>{s.title}</h3>
              <p>{s.desc}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}
