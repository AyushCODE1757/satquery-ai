import { useRef, useState } from "react";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import TaskBadge from "./TaskBadge";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const EXAMPLE_QUERIES = [
  "Describe the land-cover and major objects visible in this image.",
  "What changed between these two dates, and where did the change occur?",
  "Use the optical and SAR images together to identify built-up regions.",
];

export default function ChatBox({ onResult, demoMode }) {
  const [files, setFiles] = useState([]);
  const [query, setQuery] = useState("");
  const [log, setLog] = useState([]); // { type, message }
  const [isStreaming, setIsStreaming] = useState(false);
  const [lastTask, setLastTask] = useState(null);
  const abortRef = useRef(null);
  const logEndRef = useRef(null);

  const appendLog = (entry) => {
    setLog((prev) => [...prev, entry]);
    setTimeout(
      () => logEndRef.current?.scrollIntoView({ behavior: "smooth" }),
      0,
    );
  };

  const handleSubmit = async () => {
    if (!query.trim() || files.length === 0 || isStreaming) return;

    setLog([]);
    setLastTask(null);
    setIsStreaming(true);
    appendLog({
      type: "sys",
      message: `Uploading ${files.length} image${files.length > 1 ? "s" : ""}...`,
    });

    abortRef.current?.abort();
    abortRef.current = new AbortController();

    try {
      const formData = new FormData();
      files.forEach((f) => formData.append("images", f));
      const uploadRes = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });
      if (!uploadRes.ok) throw new Error("Upload failed");
      const { image_ids } = await uploadRes.json();

      await fetchEventSource(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, image_ids, demo_mode: demoMode }),
        signal: abortRef.current.signal,
        onmessage(ev) {
          const data = JSON.parse(ev.data);
          if (data.type === "trace")
            appendLog({ type: "trace", message: data.message });
          if (data.type === "error")
            appendLog({ type: "error", message: data.message });
          if (data.type === "final") {
            setLastTask(data.execution_summary?.task);
            appendLog({ type: "sys", message: "Execution complete." });
            onResult(data);
          }
        },
        onerror(err) {
          appendLog({ type: "error", message: err.message });
          throw err;
        },
        onclose() {
          setIsStreaming(false);
        },
      });
    } catch (err) {
      appendLog({ type: "error", message: err.message });
      setIsStreaming(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-900/60 rounded-2xl border border-slate-800/80 overflow-hidden shadow-xl shadow-black/20">
      <div className="px-4 py-3 border-b border-slate-800/80 flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-400 tracking-wide uppercase">
          Agent Trace
        </span>
        {lastTask && <TaskBadge task={lastTask} />}
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3 font-mono text-[13px] leading-relaxed space-y-1.5">
        {log.length === 0 && (
          <div className="h-full flex flex-col items-center justify-center text-center gap-3 text-slate-600 font-sans">
            <p className="text-sm">Try one of these:</p>
            <div className="flex flex-col gap-1.5 w-full max-w-xs">
              {EXAMPLE_QUERIES.map((q) => (
                <button
                  key={q}
                  onClick={() => setQuery(q)}
                  className="text-left text-xs px-3 py-2 rounded-lg border border-slate-800 text-slate-400 hover:border-cyan-800 hover:text-cyan-300 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
        {log.map((entry, i) => (
          <div
            key={i}
            className={
              entry.type === "error"
                ? "text-red-400"
                : entry.type === "sys"
                  ? "text-slate-500 italic"
                  : "text-cyan-300/90"
            }
          >
            <span className="text-slate-600 mr-1.5">
              {entry.type === "error" ? "✕" : entry.type === "sys" ? "·" : "›"}
            </span>
            {entry.message}
          </div>
        ))}
        {isStreaming && (
          <div className="flex items-center gap-1.5 text-slate-500">
            <span className="flex gap-0.5">
              <span className="w-1 h-1 rounded-full bg-cyan-400 animate-bounce [animation-delay:-0.3s]" />
              <span className="w-1 h-1 rounded-full bg-cyan-400 animate-bounce [animation-delay:-0.15s]" />
              <span className="w-1 h-1 rounded-full bg-cyan-400 animate-bounce" />
            </span>
          </div>
        )}
        <div ref={logEndRef} />
      </div>

      <div className="border-t border-slate-800/80 p-3 space-y-2 bg-slate-950/50">
        <label className="flex items-center gap-2 text-xs text-slate-400 cursor-pointer">
          <input
            type="file"
            multiple
            accept=".tif,.tiff,.png,.jpg,.jpeg"
            onChange={(e) => setFiles(Array.from(e.target.files))}
            className="text-xs text-slate-400 file:mr-2 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:bg-slate-800 file:text-slate-300 hover:file:bg-slate-700 file:cursor-pointer file:transition-colors"
          />
        </label>
        {files.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {files.map((f, i) => (
              <span
                key={i}
                className="text-[11px] px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700 flex items-center gap-1"
              >
                🛰️ {f.name}
              </span>
            ))}
          </div>
        )}
        <div className="flex gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
            placeholder="Ask about this imagery..."
            className="flex-1 bg-slate-800/80 text-slate-100 placeholder-slate-500 text-sm rounded-xl px-3.5 py-2.5 outline-none border border-slate-700 focus:border-cyan-500 focus:bg-slate-800 transition-all"
          />
          <button
            onClick={handleSubmit}
            disabled={isStreaming || !query.trim() || files.length === 0}
            className="px-4 py-2.5 rounded-xl text-sm font-medium bg-cyan-500 text-slate-950 hover:bg-cyan-400 disabled:opacity-30 disabled:cursor-not-allowed transition-all active:scale-95"
          >
            {isStreaming ? "···" : "Ask →"}
          </button>
        </div>
      </div>
    </div>
  );
}
