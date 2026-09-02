import React, { useState } from 'react';

/**
 * SatQuery AI MapOverlay Component
 * Placeholder React component demonstrating Leaflet GeoJSON layer toggles for Optical vs SAR features.
 */
export default function MapOverlay({ finalPayload }) {
  const [showOptical, setShowOptical] = useState(true);
  const [showSar, setShowSar] = useState(true);

  if (!finalPayload) {
    return (
      <div style={{ flex: 1, background: '#020617', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b', border: '1px solid #1e293b' }}>
        Leaflet Geospatial Map Overlay — Awaiting Query Results
      </div>
    );
  }

  const { geojson, confidence, execution_summary, text } = finalPayload;
  const isOpticalSarFusion = execution_summary?.task === 'optical_sar_fusion';

  const features = geojson?.features || [];
  const filteredFeatures = features.filter((feat) => {
    if (!isOpticalSarFusion) return true;
    const source = feat.properties?.source;
    if (source === 'optical' && !showOptical) return false;
    if (source === 'sar' && !showSar) return false;
    return true;
  });

  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#0f172a', borderRadius: '12px', padding: '16px', color: '#f8fafc', border: '1px solid #1e293b' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
        <h3 style={{ fontSize: '1.1rem', fontWeight: 600, color: '#38bdf8' }}>Geospatial Evidence Map</h3>
        <div style={{ display: 'flex', gap: '12px', fontSize: '0.85rem' }}>
          <span style={{ background: '#0369a1', padding: '4px 8px', borderRadius: '4px' }}>
            Task: {execution_summary?.task}
          </span>
          <span style={{ background: '#15803d', padding: '4px 8px', borderRadius: '4px' }}>
            Confidence: {(confidence * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {isOpticalSarFusion && (
        <div style={{ display: 'flex', gap: '16px', background: '#1e293b', padding: '8px 12px', borderRadius: '6px', marginBottom: '12px', fontSize: '0.85rem' }}>
          <strong style={{ color: '#f59e0b' }}>Fusion Layers:</strong>
          <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <input type="checkbox" checked={showOptical} onChange={(e) => setShowOptical(e.target.checked)} />
            Optical Layer (Sentinel-2)
          </label>
          <label style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}>
            <input type="checkbox" checked={showSar} onChange={(e) => setShowSar(e.target.checked)} />
            SAR Layer (Sentinel-1)
          </label>
        </div>
      )}

      <div style={{ background: '#020617', padding: '12px', borderRadius: '8px', marginBottom: '12px', borderLeft: '4px solid #38bdf8' }}>
        <p style={{ margin: 0, fontSize: '0.9rem', lineHeight: '1.4' }}>{text}</p>
      </div>

      {/* Simulated Leaflet Layer View */}
      <div style={{ flex: 1, minHeight: '240px', background: '#030712', borderRadius: '8px', border: '1px solid #334155', padding: '16px', display: 'flex', flexDirection: 'column', gap: '8px', overflowY: 'auto' }}>
        <div style={{ fontSize: '0.8rem', color: '#94a3b8', borderBottom: '1px solid #1e293b', pb: '4px' }}>
          Rendered GeoJSON Features ({filteredFeatures.length} active layer(s)):
        </div>
        {filteredFeatures.map((feat, idx) => (
          <div key={idx} style={{ background: feat.properties?.source === 'sar' ? '#312e81' : '#064e3b', padding: '10px', borderRadius: '6px', fontSize: '0.85rem', border: '1px solid #475569' }}>
            <div style={{ fontWeight: 600, color: '#f1f5f9' }}>
              📍 Feature {idx + 1}: {feat.properties?.label}
            </div>
            <div style={{ color: '#cbd5e1', fontSize: '0.78rem', marginTop: '4px' }}>
              Geometry Type: {feat.geometry?.type} | Confidence: {feat.properties?.confidence}
              {feat.properties?.source && ` | Source: ${feat.properties.source.toUpperCase()}`}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
