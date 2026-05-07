import React, { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Environment, ContactShadows } from '@react-three/drei';
import * as THREE from 'three';

const StressShaderMaterial = {
  uniforms: {
    uTime: { value: 0 },
    uColor1: { value: new THREE.Color('#0f172a') },
    uColor2: { value: new THREE.Color('#22c55e') },
    uColor3: { value: new THREE.Color('#f59e0b') },
    uColor4: { value: new THREE.Color('#ef4444') },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() {
      vUv = uv;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform float uTime;
    uniform vec3 uColor1;
    uniform vec3 uColor2;
    uniform vec3 uColor3;
    uniform vec3 uColor4;
    varying vec2 vUv;

    void main() {
      float d = distance(vUv, vec2(0.5, 0.5));
      float edge = 1.0 - min(min(vUv.x, 1.0-vUv.x), min(vUv.y, 1.0-vUv.y));
      float ripple = 0.08 * sin(15.0 * vUv.x + uTime * 0.6) * sin(12.0 * vUv.y);
      float stress = clamp(0.55 * (1.0 - d) + 0.5 * edge + ripple, 0.0, 1.0);

      vec3 c1 = mix(uColor1, uColor2, smoothstep(0.0, 0.35, stress));
      vec3 c2 = mix(c1, uColor3, smoothstep(0.35, 0.7, stress));
      vec3 c3 = mix(c2, uColor4, smoothstep(0.7, 1.0, stress));
      gl_FragColor = vec4(c3, 1.0);
    }
  `,
};

const HeatmapPlate = ({ length, width, thickness }) => {
  const materialRef = useRef();
  useFrame((state) => {
    if (materialRef.current) {
      materialRef.current.uniforms.uTime.value = state.clock.getElapsedTime();
    }
  });

  return (
    <mesh position={[0, thickness / 2, 0]} castShadow receiveShadow>
      <boxGeometry args={[width, thickness, length]} />
      <shaderMaterial ref={materialRef} attach="material" {...StressShaderMaterial} />
    </mesh>
  );
};

const StiffenerGrid = ({ design, plateLength, plateWidth }) => {
  const stiffenersX = [];
  const stiffenersY = [];

  if (design && (design.num_x > 0 || design.num_y > 0)) {
    const spacingXAlongLength = design.spacing_x || plateLength / (design.num_x + 1);
    for (let i = 0; i < design.num_x; i += 1) {
      const posZ = -plateLength / 2 + (i + 1) * spacingXAlongLength;
      stiffenersX.push(
        <mesh key={`x-${i}`} position={[0, -design.depth_x / 2, posZ]} castShadow>
          <boxGeometry args={[plateWidth, design.depth_x, design.width_x]} />
          <meshStandardMaterial color="#9ca3af" metalness={0.75} roughness={0.25} />
        </mesh>
      );
    }

    const spacingYAlongBreadth = design.spacing_y || plateWidth / (design.num_y + 1);
    for (let i = 0; i < design.num_y; i += 1) {
      const posX = -plateWidth / 2 + (i + 1) * spacingYAlongBreadth;
      stiffenersY.push(
        <mesh key={`y-${i}`} position={[posX, -design.depth_y / 2, 0]} castShadow>
          <boxGeometry args={[design.width_y, design.depth_y, plateLength]} />
          <meshStandardMaterial color="#a1a1aa" metalness={0.75} roughness={0.25} />
        </mesh>
      );
    }
  }

  return <group>{stiffenersX}{stiffenersY}</group>;
};

const AutoFrame = ({ plateLength, plateWidth, controlsRef }) => {
  useEffect(() => {
    if (!controlsRef.current) return;
    const maxDim = Math.max(plateLength, plateWidth, 0.3);
    const camera = controlsRef.current.object;
    const dist = maxDim * 1.6;
    camera.position.set(-dist * 0.65, dist * 0.9, dist);
    camera.near = 0.01;
    camera.far = maxDim * 25;
    camera.updateProjectionMatrix();
    controlsRef.current.target.set(0, 0, 0);
    controlsRef.current.update();
  }, [plateLength, plateWidth, controlsRef]);

  return null;
};

export default function ThreeVisualizer({ designGeometry }) {
  const controlsRef = useRef(null);
  const groupRef = useRef(null);
  const maxSource = Math.max(designGeometry?.length || 1524, designGeometry?.breadth || 762);
  const sc = 12 / maxSource;
  const plateL = (designGeometry?.length || 1524) * sc;
  const plateW = (designGeometry?.breadth || 762) * sc;

  const scaledDesign = useMemo(() => {
    if (!designGeometry) return null;
    return {
      num_x: designGeometry.num_x,
      num_y: designGeometry.num_y,
      width_x: designGeometry.width_x * sc,
      depth_x: designGeometry.depth_x * sc,
      width_y: designGeometry.width_y * sc,
      depth_y: designGeometry.depth_y * sc,
      spacing_x: (designGeometry.spacing_x || designGeometry.length / (designGeometry.num_x + 1)) * sc,
      spacing_y: (designGeometry.spacing_y || designGeometry.breadth / (designGeometry.num_y + 1)) * sc,
      thickness: designGeometry.thickness * sc,
    };
  }, [designGeometry, sc]);

  const tp = scaledDesign ? scaledDesign.thickness : 10 * sc;

  const handleExportSTL = async () => {
    if (!groupRef.current) return;
    try {
      // Scale up to true physical dimensions (mm) before exporting
      groupRef.current.scale.set(1/sc, 1/sc, 1/sc);
      groupRef.current.updateMatrixWorld(true);

      const { STLExporter } = await import('three/examples/jsm/exporters/STLExporter.js');
      const exporter = new STLExporter();
      const stlString = exporter.parse(groupRef.current);
      
      // Restore visual scale for the React Canvas
      groupRef.current.scale.set(1, 1, 1);
      groupRef.current.updateMatrixWorld(true);
      
      const blob = new Blob([stlString], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.style.display = 'none';
      link.href = url;
      link.download = `Optimized_Plate_${designGeometry.num_x}x${designGeometry.num_y}.stl`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error("Failed to export STL", err);
      alert("Failed to export 3D model. Please check the console.");
      
      // Ensure scale is restored even on failure
      groupRef.current.scale.set(1, 1, 1);
      groupRef.current.updateMatrixWorld(true);
    }
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%' }}>
      {designGeometry && (
        <button 
          onClick={handleExportSTL}
          style={{
            position: 'absolute', top: 16, right: 16, zIndex: 10,
            background: '#10b981', color: 'white', padding: '10px 20px',
            border: 'none', borderRadius: '8px', cursor: 'pointer',
            fontWeight: 'bold', fontSize: '14px',
            boxShadow: '0 4px 6px rgba(0,0,0,0.1), 0 1px 3px rgba(0,0,0,0.08)',
            display: 'flex', alignItems: 'center', gap: '8px', transition: 'all 0.2s'
          }}
          onMouseOver={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
          onMouseOut={(e) => e.currentTarget.style.transform = 'translateY(0)'}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
          Export 3D Model (.STL)
        </button>
      )}
      <Canvas shadows camera={{ position: [-8, 10, 12], fov: 42 }} gl={{ preserveDrawingBuffer: true, antialias: true }}>
        <color attach="background" args={['#F3F4F6']} />
        <ambientLight intensity={0.35} />
        <directionalLight position={[10, 20, 10]} intensity={1.6} castShadow shadow-bias={-0.0001} shadow-mapSize={[2048, 2048]} />
        <pointLight position={[-8, 6, -12]} intensity={0.45} />

        <group ref={groupRef} position={[0, 0, 0]}>
          <HeatmapPlate length={plateL} width={plateW} thickness={tp} />
          {scaledDesign && <StiffenerGrid design={scaledDesign} plateLength={plateL} plateWidth={plateW} />}
        </group>

        <AutoFrame plateLength={plateL} plateWidth={plateW} controlsRef={controlsRef} />
        <ContactShadows position={[0, -5, 0]} opacity={0.6} scale={70} blur={2.3} far={12} />
        <OrbitControls ref={controlsRef} enablePan enableZoom minDistance={1.2} maxDistance={80} minPolarAngle={Math.PI / 8} maxPolarAngle={Math.PI - Math.PI / 8} />
        <Environment preset="city" />
      </Canvas>
    </div>
  );
}

export function SurfacePlot3D({ heatmap = [], min = 0, max = 1, title="3D Surface", unit="" }) {
  const geometry = useMemo(() => {
    if (!heatmap || !heatmap.length) return null;
    const rows = heatmap.length;
    const cols = heatmap[0].length;
    const geo = new THREE.PlaneGeometry(10, 10, cols - 1, rows - 1);
    const positions = geo.attributes.position;
    const colors = [];
    const color = new THREE.Color();
    
    for (let i = 0; i < positions.count; i++) {
      const x = i % cols;
      const y = Math.floor(i / cols);
      const val = heatmap[rows - 1 - y]?.[x] || 0;
      
      const t = (val - min) / Math.max(max - min, 1e-9);
      
      // Blue (240) -> Green (120) -> Red (0)
      const hue = (1 - t) * 240 / 360;
      color.setHSL(hue, 1.0, 0.5);
      
      colors.push(color.r, color.g, color.b);
      positions.setZ(i, t * 2.5);
    }
    
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geo.computeVertexNormals();
    return geo;
  }, [heatmap, min, max]);

  if (!geometry) return null;

  return (
    <div className="plot-wrap" style={{ position: 'relative', height: '350px', padding: 0, overflow: 'hidden', border: '1px solid var(--border-color)', borderRadius: '8px', display: 'flex', flexDirection: 'column' }}>
      <div className="hm-head" style={{ position: 'absolute', top: '16px', left: '16px', zIndex: 10, margin: 0 }}>
        <span style={{ fontWeight: 600, color: '#111827', backgroundColor: 'rgba(255, 255, 255, 0.9)', padding: '6px 12px', borderRadius: '6px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>{title}</span>
      </div>
      
      <div style={{ position: 'absolute', bottom: '16px', left: '16px', right: '16px', zIndex: 10, backgroundColor: 'rgba(255, 255, 255, 0.9)', padding: '8px 12px', borderRadius: '6px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', fontWeight: 600, color: '#374151', marginBottom: '4px' }}>
          <span>{min.toFixed(2)}{unit}</span>
          <span>{((min + max) / 2).toFixed(2)}{unit}</span>
          <span>{max.toFixed(2)}{unit}</span>
        </div>
        <div style={{ height: '8px', width: '100%', borderRadius: '4px', background: 'linear-gradient(to right, hsl(240, 100%, 50%), hsl(120, 100%, 50%), hsl(0, 100%, 50%))' }} />
      </div>

      <Canvas camera={{ position: [0, -8, 8], fov: 45 }} gl={{ antialias: true }}>
        <color attach="background" args={['#FFFFFF']} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[10, 10, 10]} intensity={1.5} />
        <mesh geometry={geometry} rotation={[-Math.PI / 2, 0, 0]}>
          <meshStandardMaterial vertexColors side={THREE.DoubleSide} wireframe={false} roughness={0.3} metalness={0.2} />
        </mesh>
        <OrbitControls autoRotate autoRotateSpeed={1.5} enableZoom={true} />
      </Canvas>
    </div>
  );
}
