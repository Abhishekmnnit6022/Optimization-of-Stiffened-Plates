import React, { useMemo, useState } from 'react';
import axios from 'axios';
import ThreeVisualizer, { SurfacePlot3D } from './ThreeVisualizer';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend, PieChart, Pie, Cell } from 'recharts';
import './index.css';

const CASES = {
  transverse: { bc: 0, plateLength: 2200, plateBreadth: 420, initialThickness: 12, uniformLoad: 22 },
  longitudinal: { bc: 0, plateLength: 620, plateBreadth: 1900, initialThickness: 12, uniformLoad: 22 },
  grid: { bc: 1, plateLength: 1400, plateBreadth: 1200, initialThickness: 16, uniformLoad: 45 },
};

function ParetoScatterChart({ data }) {
  if (!data || !data.length) return null;
  return (
    <div className="plot-wrap">
      <div className="hm-head"><span>Pareto Front (Deflection vs Stress)</span></div>
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis type="number" dataKey="deflection" name="Deflection" unit=" mm" domain={['auto', 'auto']} tick={{fontSize: 12}} />
            <YAxis type="number" dataKey="stress" name="Stress" unit=" MPa" domain={['auto', 'auto']} tick={{fontSize: 12}} />
            <Tooltip cursor={{ strokeDasharray: '3 3' }} />
            <Scatter name="Population" data={data} fill="#3b82f6" />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function CandidatesRadarChart({ candidates }) {
  if (!candidates || candidates.length < 4) return null;
  
  // Normalize values for radar to specifically highlight the DIFFERENCES (min-max scaling)
  const normalize = (vals, inverse=false) => {
    const min = Math.min(...vals);
    const max = Math.max(...vals);
    if (max - min < 1e-9) return vals.map(() => 0.5);
    return vals.map(v => inverse ? 1 - ((v - min) / (max - min)) : (v - min) / (max - min));
  };

  const defs = normalize(candidates.map(c => c.deflection), true); // Inverse: lower deflection is "better/bigger" on radar
  const strs = normalize(candidates.map(c => c.stress), true);
  const freqs = normalize(candidates.map(c => c.frequency), false); // higher frequency is better
  const vols = normalize(candidates.map(c => c.stiffener_volume), true); // lower volume is better

  const data = [
    { subject: 'Stiffness (Low Defl)', A: defs[0], B: defs[1], C: defs[2], D: defs[3] },
    { subject: 'Safety (Low Stress)', A: strs[0], B: strs[1], C: strs[2], D: strs[3] },
    { subject: 'Frequency (High Hz)', A: freqs[0], B: freqs[1], C: freqs[2], D: freqs[3] },
    { subject: 'Lightness (Low Vol)', A: vols[0], B: vols[1], C: vols[2], D: vols[3] },
  ];
  return (
    <div className="plot-wrap">
      <div className="hm-head"><span>Trade-off Radar (Normalized Variance)</span></div>
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <RadarChart cx="50%" cy="50%" outerRadius="65%" data={data}>
            <PolarGrid />
            <PolarAngleAxis dataKey="subject" tick={{fontSize: 11, fill: '#6b7280'}} />
            <PolarRadiusAxis angle={30} domain={[0, 1]} tick={false} />
            <Radar name="Rank 1 (Best)" dataKey="A" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
            <Radar name="Rank 2 (Strong)" dataKey="B" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} />
            <Radar name="Rank 3 (Safe)" dataKey="C" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.3} />
            <Radar name="Rank 4 (Light)" dataKey="D" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.3} />
            <Legend wrapperStyle={{fontSize: '12px'}} />
            <Tooltip />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function VolumePieChart({ geometry }) {
  if (!geometry) return null;
  const data = [
    { name: 'Stiffener Vol', value: geometry.stiffener_volume },
    { name: 'Plate Vol', value: geometry.plate_volume_remaining },
  ];
  const COLORS = ['#3b82f6', '#cbd5e1'];
  return (
    <div className="plot-wrap">
      <div className="hm-head"><span>Volume Distribution</span></div>
      <div style={{ width: '100%', height: 300 }}>
        <ResponsiveContainer>
          <PieChart>
            <Pie data={data} cx="50%" cy="50%" innerRadius={60} outerRadius={80} fill="#8884d8" paddingAngle={5} dataKey="value" label={({name, percent}) => `${name} ${(percent * 100).toFixed(0)}%`}>
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function App() {
  const [plateLength, setPlateLength] = useState(1524);
  const [plateBreadth, setPlateBreadth] = useState(762);
  const [initialThickness, setInitialThickness] = useState(12);
  const [uniformLoad, setUniformLoad] = useState(10.0);
  const [bc, setBc] = useState(1);
  const [geometry, setGeometry] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const applyCase = (preset) => {
    setBc(preset.bc);
    setPlateLength(preset.plateLength);
    setPlateBreadth(preset.plateBreadth);
    setInitialThickness(preset.initialThickness);
    setUniformLoad(preset.uniformLoad);
  };

  const handleOptimize = async () => {
    setLoading(true);
    setError('');
    try {
      const resp = await axios.post('http://127.0.0.1:8000/optimize', {
        bc,
        plate_length: plateLength,
        plate_breadth: plateBreadth,
        initial_thickness: initialThickness,
        q_load: uniformLoad,
      });
      setGeometry(resp.data.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Could not get result from backend.');
    } finally {
      setLoading(false);
    }
  };

  const heatmapStats = useMemo(() => {
    if (!geometry) return { min: 0, max: 1 };
    return { min: geometry.heatmap_min ?? 0, max: geometry.heatmap_max ?? 1 };
  }, [geometry]);

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="header">
          <h1>Plate Stiffener Optimizer</h1>
          <p>Physics-guided layout and thickness optimization</p>
        </div>

        <div className="control-group">
          <label>Boundary Condition</label>
          <select value={bc} onChange={(e) => setBc(parseInt(e.target.value, 10))}>
            <option value={1}>Fixed (Clamped)</option>
            <option value={0}>Simply Supported</option>
          </select>
        </div>
        <div className="control-group">
          <label>Plate Length (mm)</label>
          <input type="number" min="100" step="10" value={plateLength} onChange={(e) => setPlateLength(parseFloat(e.target.value) || 0)} />
        </div>
        <div className="control-group">
          <label>Plate Breadth (mm)</label>
          <input type="number" min="100" step="10" value={plateBreadth} onChange={(e) => setPlateBreadth(parseFloat(e.target.value) || 0)} />
        </div>
        <div className="control-group">
          <label>Initial Plate Thickness (mm)</label>
          <input type="number" min="2" step="0.5" value={initialThickness} onChange={(e) => setInitialThickness(parseFloat(e.target.value) || 0)} />
        </div>
        <div className="control-group">
          <label>Uniform Load q (N/mm^2)</label>
          <input type="number" min="0.1" step="0.5" value={uniformLoad} onChange={(e) => setUniformLoad(parseFloat(e.target.value) || 0)} />
        </div>

        <button className="btn-optimize" onClick={handleOptimize} disabled={loading}>
          {loading ? 'Optimizing...' : 'Calculate Optimal Topology'}
        </button>

        {error && <div className="error-box">{error}</div>}

        {geometry && !loading && (
          <div className="results-panel">
            <h3>Optimization Results</h3>
            <div className="result-grid">
              <div className="result-row"><span className="r-label">Pattern</span><span className="r-val">{geometry.pattern_type.toUpperCase()}</span></div>
              <div className="result-row"><span className="r-label">X / Y Stiffeners</span><span className="r-val">{geometry.num_x} / {geometry.num_y}</span></div>
              <div className="result-row"><span className="r-label">Thickness (new)</span><span className="r-val">{geometry.thickness.toFixed(2)} mm</span></div>
              <div className="result-row"><span className="r-label">Deflection</span><span className="r-val">{geometry.optimal_deflection.toFixed(4)} mm</span></div>
              <div className="result-row"><span className="r-label">Stress</span><span className="r-val">{geometry.optimal_stress.toFixed(3)}</span></div>
              <div className="result-row"><span className="r-label">Frequency</span><span className="r-val">{geometry.optimal_frequency.toFixed(2)} Hz</span></div>
              <div className="result-row"><span className="r-label">Angle X / Y</span><span className="r-val">{geometry.angle_x_deg.toFixed(1)} deg / {geometry.angle_y_deg.toFixed(1)} deg</span></div>
            </div>

          </div>
        )}
      </aside>

      <main className="canvas-wrapper">
        <div className="main-3d-view">
          <ThreeVisualizer designGeometry={geometry} />
        </div>
        {geometry && !loading && (
          <section className="main-analysis">
            <h3>Comparative Analysis</h3>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <SurfacePlot3D heatmap={geometry.deflection_heatmap} min={0} max={geometry.def_max} title="Physical Deflection Surface" unit=" mm" />
              <SurfacePlot3D heatmap={geometry.stress_heatmap} min={0} max={geometry.str_max} title="Physical Stress Surface" unit=" MPa" />
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '16px', marginTop: '16px' }}>
              <ParetoScatterChart data={geometry.pareto_points || []} />
              <CandidatesRadarChart candidates={geometry.candidates || []} />
              <VolumePieChart geometry={geometry} />
            </div>
            <div className="alt-grid main-alt-grid">
              {(geometry.candidates || []).map((c, idx) => (
                <div key={c.rank} className={`alt-card ${idx === 0 ? 'optimal-card' : ''}`}>
                  <div className="alt-title">
                    Rank {c.rank} - {c.pattern_type.toUpperCase()}
                    {idx === 0 && <span className="badge-optimal">OPTIMAL</span>}
                  </div>
                  <div className="alt-row"><span>Grid (nx/ny)</span> <strong>{c.num_x}/{c.num_y}</strong></div>
                  <div className="alt-row"><span>Angles</span> <strong>{c.angle_x_deg.toFixed(1)}° / {c.angle_y_deg.toFixed(1)}°</strong></div>
                  <div className="alt-row"><span>Depth (x/y)</span> <strong>{c.depth_x.toFixed(1)} / {c.depth_y.toFixed(1)} mm</strong></div>
                  <div className="alt-row"><span>Width (x/y)</span> <strong>{c.width_x.toFixed(1)} / {c.width_y.toFixed(1)} mm</strong></div>
                  <div className="alt-row"><span>Spacing (x/y)</span> <strong>{c.spacing_x.toFixed(1)} / {c.spacing_y.toFixed(1)} mm</strong></div>
                  <div className="alt-row"><span>Length (x/y)</span> <strong>{geometry.stiffener_length_x.toFixed(1)} / {geometry.stiffener_length_y.toFixed(1)} mm</strong></div>
                  <div className="alt-row"><span>Thickness</span> <strong>{c.thickness.toFixed(2)} mm</strong></div>
                  <div className="alt-row"><span>Deflection</span> <strong>{c.deflection.toFixed(4)} mm</strong></div>
                  <div className="alt-row"><span>Stress</span> <strong>{c.stress.toFixed(3)}</strong></div>
                  <div className="alt-row"><span>Frequency</span> <strong>{c.frequency.toFixed(2)} Hz</strong></div>
                  <div className="alt-row"><span>Volume (S/P)</span> <strong>{c.stiffener_volume.toFixed(0)} / {c.plate_volume_remaining.toFixed(0)}</strong></div>
                </div>
              ))}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}
