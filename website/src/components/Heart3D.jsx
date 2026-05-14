import { useRef, useEffect, useMemo, Suspense } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { useGLTF, Float, ContactShadows, Environment } from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'

// ── Anatomical colour map ────────────────────────────────────────────────────
// Left heart (oxygenated) = reds / crimsons
// Right heart (deoxygenated) = blue-purples
// Great vessels follow the same rule
const STRUCTURE_COLORS = {
  LV_free:          { color: 0xb91c1c, emissive: 0x7f1d1d },   // deep crimson
  LV_anterior:      { color: 0xdc2626, emissive: 0x991b1b },
  LV_lateral:       { color: 0xef4444, emissive: 0xb91c1c },
  LV_inferior:      { color: 0xdc2626, emissive: 0x991b1b },
  LV_septal:        { color: 0xc81e1e, emissive: 0x7f1d1d },
  LA_lateral:       { color: 0xf87171, emissive: 0xdc2626 },   // lighter — atria
  LA_superior:      { color: 0xfca5a5, emissive: 0xef4444 },
  LA_auricle:       { color: 0xfb7185, emissive: 0xe11d48 },
  RV_outflow:       { color: 0x4f46e5, emissive: 0x3730a3 },   // indigo-blue
  RV_inflow:        { color: 0x6366f1, emissive: 0x4338ca },
  RA_anterior:      { color: 0x818cf8, emissive: 0x6366f1 },   // periwinkle
  RA_lateral:       { color: 0x93c5fd, emissive: 0x60a5fa },   // pale blue
  RA_auricle:       { color: 0xa5b4fc, emissive: 0x818cf8 },
  Aorta_ascending:  { color: 0xff2222, emissive: 0xcc0000 },   // vivid red
  Aorta_bulb:       { color: 0xff3333, emissive: 0xcc0000 },
  Pulmonary_trunk:  { color: 0x7c3aed, emissive: 0x5b21b6 },   // violet
}

function lubDub(t) {
  const phase = (t % (60 / 72)) / (60 / 72)
  if (phase < 0.07)                  return Math.sin(phase / 0.07 * Math.PI) * 0.08
  if (phase > 0.13 && phase < 0.22)  return Math.sin((phase - 0.13) / 0.09 * Math.PI) * 0.04
  return 0
}

// Cache materials so we only create them once
const MAT_CACHE = {}
function getMat(name) {
  if (!MAT_CACHE[name]) {
    const c = STRUCTURE_COLORS[name] ?? { color: 0xb91c1c, emissive: 0x7f1d1d }
    MAT_CACHE[name] = new THREE.MeshPhysicalMaterial({
      color:              c.color,
      emissive:           c.emissive,
      emissiveIntensity:  0.22,
      roughness:          0.45,
      metalness:          0.04,
      clearcoat:          0.55,
      clearcoatRoughness: 0.3,
    })
  }
  return MAT_CACHE[name]
}

function BeatLight() {
  const ref = useRef()
  useFrame(({ clock }) => {
    if (!ref.current) return
    ref.current.intensity = 2 + lubDub(clock.getElapsedTime()) * 14
  })
  return <pointLight ref={ref} position={[0, 0.5, 3]} color="#ff2222" intensity={2} />
}

function AnatomicalHeart() {
  const { scene } = useGLTF('/heart.glb')
  const animRef   = useRef()

  const { normScale, centreOffset } = useMemo(() => {
    const box  = new THREE.Box3().setFromObject(scene)
    const size = new THREE.Vector3()
    const ctr  = new THREE.Vector3()
    box.getSize(size)
    box.getCenter(ctr)
    return {
      normScale:    3.0 / Math.max(size.x, size.y, size.z),
      centreOffset: ctr.negate(),
    }
  }, [scene])

  useEffect(() => {
    scene.traverse(child => {
      if (!child.isMesh) return
      // Keep the original Sketchfab materials (textures, normal maps, etc.)
      child.castShadow = true
    })
  }, [scene])

  useFrame(({ clock }) => {
    if (!animRef.current) return
    const t    = clock.getElapsedTime()
    const beat = lubDub(t)
    animRef.current.rotation.y = t * 0.20
    animRef.current.rotation.z = Math.sin(t * 0.3) * 0.04
    animRef.current.scale.setScalar(normScale * (1 + beat))
  })

  return (
    <group ref={animRef}>
      <group position={[centreOffset.x, centreOffset.y, centreOffset.z]}>
        <primitive object={scene} />
      </group>
    </group>
  )
}

function LoadingHeart() {
  const ref = useRef()
  useFrame(({ clock }) => {
    if (!ref.current) return
    const t = clock.getElapsedTime()
    ref.current.scale.setScalar(1.3 + lubDub(t) * 1.4)
    ref.current.rotation.y += 0.006
  })
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[1, 32, 32]} />
      <meshPhysicalMaterial color={0xb91c1c} roughness={0.42} emissive={0x7f1d1d} emissiveIntensity={0.3} clearcoat={0.5} />
    </mesh>
  )
}

export default function Heart3D() {
  return (
    <Canvas
      style={{ width: '100%', height: '100%' }}
      camera={{ position: [0, 0, 5.5], fov: 50 }}
      gl={{ antialias: true, alpha: true }}
    >
      <ambientLight intensity={0.8} />
      <pointLight position={[-5,  6, 3]} intensity={2.5} color="#ffffff" />
      <pointLight position={[ 5, -2, 3]} intensity={1.5} color="#ffffff" />
      <pointLight position={[ 0, -5, 4]} intensity={1.0} color="#ffffff" />
      <BeatLight />

      <Float speed={1.2} rotationIntensity={0} floatIntensity={0.5}>
        <Suspense fallback={<LoadingHeart />}>
          <AnatomicalHeart />
        </Suspense>
      </Float>

      <ContactShadows position={[0, -2.8, 0]} opacity={0.3} scale={10} blur={3} color="#660000" />

      <EffectComposer>
        <Bloom intensity={1.2} luminanceThreshold={0.15} luminanceSmoothing={0.85} mipmapBlur />
      </EffectComposer>

      <Environment preset="studio" />
    </Canvas>
  )
}

useGLTF.preload('/heart.glb')
