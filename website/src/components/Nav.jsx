import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

export default function Nav() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 40)
    window.addEventListener('scroll', fn)
    return () => window.removeEventListener('scroll', fn)
  }, [])

  return (
    <motion.nav
      className={`nav${scrolled ? ' nav--scrolled' : ''}`}
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.65, ease: 'easeOut' }}
    >
      <a href="#" className="nav-logo">
        <span className="nav-heart">♥</span>
        <span>FetalEcho<span className="gradient-text"> Assess</span></span>
      </a>
      <div className="nav-links">
        <a href="#how-it-works" className="nav-link">Workflow</a>
        <a href="#demo" className="nav-link">Assessment</a>
        <a href="#demo" className="btn-primary">Open Tool →</a>
      </div>
    </motion.nav>
  )
}
