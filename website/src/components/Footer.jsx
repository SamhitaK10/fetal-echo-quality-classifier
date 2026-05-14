export default function Footer() {
  return (
    <footer className="footer">
      <div className="footer-logo">
        <span className="footer-heart">♥</span>
        <span>FetalEcho<span className="gradient-text"> Assess</span></span>
      </div>
      <p className="footer-copy">
        First-trimester fetal cardiac quality classification · EfficientNet-B3 + EigenCAM
      </p>
      <div className="footer-links">
        <a href="https://huggingface.co/spaces/samhitak10/fetal-echo-quality-demo" className="footer-link" target="_blank" rel="noopener noreferrer">HF Space</a>
        <a href="https://huggingface.co/samhitak10/fetal-echo-quality-classifier" className="footer-link" target="_blank" rel="noopener noreferrer">Model</a>
      </div>
    </footer>
  )
}
