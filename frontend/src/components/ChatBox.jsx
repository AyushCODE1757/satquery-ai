import React, { useState } from 'react';

/**
 * SatQuery AI ChatBox Component
 * Handles query submission, demo_mode toggle, file upload, and streaming agent traces.
 */
export default function ChatBox({ onQueryResult, API_BASE = 'http://localhost:8000' }) {
  const [query, setQuery] = useState('');
  const [imageIds, setImageIds] = useState([]);
  const [demoMode, setDemoMode] = useState(false);
  const [traces, setTraces] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const handleFileUpload = async (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    if (files.length > 2) {
      alert('You can upload a maximum of 2 satellite images (e.g. T1/T2 or Optical/SAR).');
      return;
    }

    setIsUploading(true);
    setErrorMsg('');
    const formData = new FormData();
    files.forEach((file) => formData.append('images', file));

    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) throw new Error(`Upload failed with status ${res.status}`);
      const data = await res.json();
      setImageIds(data.image_ids);
      setTraces((prev) => [...prev, `[System] Uploaded ${data.image_ids.length} image(s): ${data.image_ids.join(', ')}`]);
    } catch (err) {
      setErrorMsg(`Upload error: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsQuerying(true);
    setTraces([]);
    setErrorMsg('');

    try {
      const response = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: query.trim(),
          image_ids: imageIds,
          demo_mode: demoMode,
        }),
      });

      if (!response.ok) throw new Error(`Query endpoint returned HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;
            try {
              const event = JSON.parse(jsonStr);
              if (event.type === 'trace') {
                setTraces((prev) => [...prev, event.message]);
              } else if (event.type === 'final') {
                if (onQueryResult) onQueryResult(event);
              } else if (event.type === 'error') {
                setErrorMsg(event.message);
                setTraces((prev) => [...prev, `[ERROR] ${event.message}`]);
              }
            } catch (pErr) {
              console.error('Failed to parse SSE line:', line, pErr);
            }
          }
        }
      }
    } catch (err) {
      setErrorMsg(`Connection error: ${err.message}`);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', gap: '16px', padding: '16px', background: '#0f172a', color: '#f8fafc', borderRadius: '12px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <h2 style={{ fontSize: '1.25rem', fontWeight: 600, color: '#38bdf8' }}>SatQuery AI Assistant</h2>
        <label style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontSize: '0.875rem' }}>
          <input
            type="checkbox"
            checked={demoMode}
            onChange={(e) => setDemoMode(e.target.checked)}
            style={{ width: '16px', height: '16px' }}
          />
          <span style={{ color: demoMode ? '#f59e0b' : '#94a3b8', fontWeight: demoMode ? 'bold' : 'normal' }}>
            {demoMode ? '⚡ Demo Mode (Pitch Canned Fallback)' : '🤖 Model Inference Mode'}
          </span>
        </label>
      </div>

      <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
        <input
          type="file"
          multiple
          accept="image/*,.tif,.tiff"
          onChange={handleFileUpload}
          disabled={isUploading}
          style={{ fontSize: '0.875rem' }}
        />
        {isUploading && <span style={{ fontSize: '0.85rem', color: '#38bdf8' }}>Extracting CRS & Bounds...</span>}
      </div>

      {imageIds.length > 0 && (
        <div style={{ fontSize: '0.8rem', color: '#4ade80' }}>
          Active Image IDs: {imageIds.join(', ')}
        </div>
      )}

      {/* Streaming Agent Log Trace Panel */}
      <div style={{ flex: 1, minHeight: '180px', maxHeight: '300px', overflowY: 'auto', background: '#020617', padding: '12px', borderRadius: '8px', fontFamily: 'monospace', fontSize: '0.85rem', border: '1px solid #1e293b' }}>
        <div style={{ color: '#64748b', marginBottom: '6px' }}>--- Agent Execution Trace ---</div>
        {traces.map((msg, idx) => (
          <div key={idx} style={{ color: msg.startsWith('[ERROR]') ? '#ef4444' : msg.startsWith('Routed') ? '#38bdf8' : '#e2e8f0', marginBottom: '4px' }}>
            ➔ {msg}
          </div>
        ))}
        {isQuerying && <div style={{ color: '#f59e0b' }}>⌛ Processing query...</div>}
      </div>

      {errorMsg && <div style={{ color: '#ef4444', fontSize: '0.85rem' }}>{errorMsg}</div>}

      <form onSubmit={handleQuerySubmit} style={{ display: 'flex', gap: '8px' }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a satellite imagery question (e.g. 'Detect bi-temporal changes in sector A')..."
          disabled={isQuerying}
          style={{ flex: 1, padding: '10px 14px', borderRadius: '6px', border: '1px solid #334155', background: '#1e293b', color: '#fff', fontSize: '0.9rem' }}
        />
        <button
          type="submit"
          disabled={isQuerying || !query.trim()}
          style={{ padding: '10px 20px', borderRadius: '6px', border: 'none', background: '#0284c7', color: '#fff', fontWeight: 600, cursor: 'pointer' }}
        >
          {isQuerying ? 'Querying...' : 'Ask Agent'}
        </button>
      </form>
    </div>
  );
}
