export default function ConfidenceBar({ value }) {
  if (value === undefined) return null;
  const pct = Math.round(value * 100);
  const color =
    pct >= 80 ? "bg-emerald-400" : pct >= 50 ? "bg-amber-400" : "bg-red-400";
  return (
    <div className="flex items-center gap-1.5 w-24">
      <div className="flex-1 h-1 rounded-full bg-slate-800 overflow-hidden">
        <div
          className={`h-full ${color} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-[10px] text-slate-500 tabular-nums">{pct}%</span>
    </div>
  );
}
