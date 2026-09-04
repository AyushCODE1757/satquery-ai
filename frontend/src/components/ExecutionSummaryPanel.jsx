import { useState } from "react";

export default function ExecutionSummaryPanel({ executionSummary }) {
  const [expanded, setExpanded] = useState(false);
  if (!executionSummary) return null;

  const { task, models_used = [], params = {} } = executionSummary;

  return (
    <div className="border-t border-slate-800/80">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between px-3 py-2 text-[11px] text-slate-500 hover:text-slate-300 transition-colors"
      >
        <span className="font-medium tracking-wide uppercase">
          Execution Summary
        </span>
        <span className="text-slate-600">{expanded ? "▲" : "▼"}</span>
      </button>
      {expanded && (
        <div className="px-3 pb-3 space-y-2 text-[12px] font-mono">
          <div className="flex gap-2">
            <span className="text-slate-500 w-20 shrink-0">task</span>
            <span className="text-cyan-300">{task}</span>
          </div>
          <div className="flex gap-2">
            <span className="text-slate-500 w-20 shrink-0">models</span>
            <span className="text-slate-300">
              {models_used.join(", ") || "—"}
            </span>
          </div>
          {Object.entries(params).length > 0 && (
            <div className="flex gap-2">
              <span className="text-slate-500 w-20 shrink-0">params</span>
              <pre className="text-slate-400 whitespace-pre-wrap break-all">
                {JSON.stringify(params, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
