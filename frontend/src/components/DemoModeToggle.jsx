export default function DemoModeToggle({ enabled, onChange }) {
  return (
    <button
      onClick={() => onChange(!enabled)}
      className="group flex items-center gap-2 pl-2.5 pr-1 py-1 rounded-full border border-slate-800 bg-slate-900/60 hover:border-slate-700 transition-colors"
      title="Toggle demo-mode fallback (canned response path)"
    >
      <span
        className={`text-[11px] font-medium tracking-wide ${enabled ? "text-amber-400" : "text-slate-500"}`}
      >
        {enabled ? "DEMO MODE" : "LIVE INFERENCE"}
      </span>
      <span
        className={`relative w-8 h-4.5 rounded-full transition-colors duration-200 ${
          enabled ? "bg-amber-500" : "bg-slate-700"
        }`}
        style={{ height: "18px" }}
      >
        <span
          className={`absolute top-0.5 left-0.5 h-3.5 w-3.5 rounded-full bg-white shadow transition-transform duration-200 ${
            enabled ? "translate-x-3.5" : "translate-x-0"
          }`}
        />
      </span>
    </button>
  );
}
