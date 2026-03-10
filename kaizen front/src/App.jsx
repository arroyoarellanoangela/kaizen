import { useState, useRef, useEffect, useCallback } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './index.css';

const API = '/api';

function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      role: 'bot',
      content: 'A mind at rest moves further. How can I assist you with your knowledge today?',
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const chatEndRef = useRef(null);

  // Sidebar state
  const [status, setStatus] = useState(null);
  const [topK, setTopK] = useState(5);
  const [category, setCategory] = useState('');
  const [isIngesting, setIsIngesting] = useState(false);
  const [ingestResult, setIngestResult] = useState(null);

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Poll status every 5s
  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API}/status`);
      if (res.ok) setStatus(await res.json());
    } catch { /* backend not running yet */ }
  }, []);

  useEffect(() => {
    fetchStatus();
    const id = setInterval(fetchStatus, 5000);
    return () => clearInterval(id);
  }, [fetchStatus]);

  // Send query with SSE streaming
  const handleSend = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const query = inputValue.trim();
    const userMsg = { id: Date.now(), role: 'user', content: query };
    setMessages((prev) => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    // Placeholder bot message that we'll stream into
    const botId = Date.now() + 1;
    setMessages((prev) => [...prev, { id: botId, role: 'bot', content: '', sources: [] }]);

    try {
      const res = await fetch(`${API}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, top_k: topK, category: category || null }),
      });

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep incomplete line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const data = JSON.parse(line.slice(6));

          if (data.type === 'sources') {
            setMessages((prev) =>
              prev.map((m) => (m.id === botId ? { ...m, sources: data.sources } : m))
            );
          } else if (data.type === 'token') {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === botId ? { ...m, content: m.content + data.content } : m
              )
            );
          }
        }
      }
    } catch (err) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botId
            ? { ...m, content: `Connection error: ${err.message}. Is the API running?` }
            : m
        )
      );
    }

    setIsLoading(false);
  };

  // Ingest
  const handleIngest = async (force = false) => {
    setIsIngesting(true);
    setIngestResult(null);
    try {
      const res = await fetch(`${API}/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ force }),
      });
      const data = await res.json();
      fetchStatus();
      setIngestResult({ ok: true, msg: `${data.added} added, ${data.skipped} skipped` });
    } catch (err) {
      setIngestResult({ ok: false, msg: err.message });
    }
    setIsIngesting(false);
    setTimeout(() => setIngestResult(null), 6000);
  };

  const gpu = status?.gpu;
  const connected = status !== null;

  return (
    <div className="layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <a href="#" className="brand-logo">Kaizen<em>.</em></a>
        </div>

        {/* System Status */}
        <div className="metrics-section">
          <div className="metrics-title">System</div>

          <div className="metric-group">
            <div className="metric-row">
              <span className="metric-label">Status</span>
              <span className="metric-value status-value">
                <span className={`status-dot ${connected ? 'online' : 'offline'}`} />
                {connected ? 'Online' : 'Offline'}
              </span>
            </div>
            <div className="metric-row">
              <span className="metric-label">LLM</span>
              <span className="metric-value">{status?.llm_model || '—'}</span>
            </div>
            <div className="metric-row">
              <span className="metric-label">Pipeline</span>
              <span className="metric-value">FP16 Dual-Stage</span>
            </div>
            <div className="metric-row">
              <span className="metric-label">Chunks</span>
              <span className="metric-value">
                {status ? status.chunks.toLocaleString() : '—'}
              </span>
            </div>
          </div>
        </div>

        {/* GPU Telemetry */}
        {gpu && (
          <div className="metrics-section">
            <div className="metrics-title">GPU</div>

            <div className="metric-group">
              <div className="metric-row">
                <span className="metric-label">Device</span>
                <span className="metric-value gpu-device">{gpu.name}</span>
              </div>
            </div>

            <div className="gpu-stats">
              <div className="gpu-stat">
                <div className="gpu-stat-header">
                  <span className="metric-label">VRAM</span>
                  <span className="gpu-stat-value">{gpu.vram_used_gb.toFixed(1)} / {gpu.vram_total_gb.toFixed(1)} GB</span>
                </div>
                <div className="progress-bg">
                  <div
                    className={`progress-fill ${gpu.vram_pct > 85 ? 'critical' : gpu.vram_pct > 65 ? 'warn' : ''}`}
                    style={{ width: `${gpu.vram_pct}%` }}
                  />
                </div>
              </div>

              <div className="gpu-stat">
                <div className="gpu-stat-header">
                  <span className="metric-label">Compute</span>
                  <span className="gpu-stat-value">{gpu.gpu_util}%</span>
                </div>
                <div className="progress-bg">
                  <div
                    className={`progress-fill ${gpu.gpu_util > 85 ? 'critical' : gpu.gpu_util > 65 ? 'warn' : ''}`}
                    style={{ width: `${gpu.gpu_util}%` }}
                  />
                </div>
              </div>

              <div className="gpu-stat">
                <div className="gpu-stat-header">
                  <span className="metric-label">Temp</span>
                  <span className={`gpu-stat-value ${gpu.temp_c > 80 ? 'temp-hot' : gpu.temp_c > 65 ? 'temp-warm' : ''}`}>
                    {gpu.temp_c}°C
                  </span>
                </div>
                <div className="progress-bg">
                  <div
                    className={`progress-fill ${gpu.temp_c > 80 ? 'critical' : gpu.temp_c > 65 ? 'warn' : ''}`}
                    style={{ width: `${Math.min(gpu.temp_c, 100)}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Query Controls */}
        <div className="metrics-section">
          <div className="metrics-title">Controls</div>

          <div className="metric-group">
            <div className="metric-row">
              <span className="metric-label">Results</span>
              <span className="metric-value">{topK}</span>
            </div>
            <input
              type="range"
              min="1"
              max="15"
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="sidebar-range"
            />
          </div>

          <div className="metric-group">
            <label className="metric-label control-label">Category Filter</label>
            <input
              type="text"
              className="sidebar-input"
              placeholder="e.g. ai, data-engineering"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            />
          </div>

          <button
            className="sidebar-btn"
            onClick={() => handleIngest(false)}
            disabled={isIngesting}
          >
            {isIngesting ? 'Ingesting...' : 'Ingest Now'}
          </button>

          {ingestResult && (
            <div className={`ingest-toast ${ingestResult.ok ? 'ok' : 'err'}`}>
              {ingestResult.msg}
            </div>
          )}
        </div>

        <div className="brand-kanji">改善</div>
      </aside>

      {/* Main chat */}
      <main className="main-content">
        <div className="chat-container">
          {messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.role}`}>
              {msg.role === 'bot' ? (
                <>
                  <div className="markdown-body">
                    <Markdown remarkPlugins={[remarkGfm]}>
                      {msg.content || (isLoading ? '...' : '')}
                    </Markdown>
                  </div>
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-list">
                      {msg.sources.map((src, i) => (
                        <details key={i} className="source-item">
                          <summary>
                            <span className="source-path">
                              {src.category}/{src.source}
                            </span>
                            <span className="source-score">
                              {(src.score * 100).toFixed(0)}%
                            </span>
                          </summary>
                          <div className="source-text">{src.text}</div>
                        </details>
                      ))}
                    </div>
                  )}
                </>
              ) : (
                msg.content
              )}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        <div className="input-container">
          <form className="input-box" onSubmit={handleSend}>
            <input
              type="text"
              placeholder="Query the knowledge base..."
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              disabled={isLoading}
              autoFocus
            />
            <button type="submit" disabled={isLoading}>
              {isLoading ? '...' : 'Submit'}
            </button>
          </form>
        </div>
      </main>
    </div>
  );
}

export default App;
