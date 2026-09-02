import {
  ReactCompareSlider,
  ReactCompareSliderImage,
} from "react-compare-slider";

export default function ChangeCompare({ beforeUrl, afterUrl }) {
  if (!beforeUrl || !afterUrl) return null;

  return (
    <div className="rounded-2xl border border-slate-800/80 overflow-hidden bg-slate-900/60 shadow-xl shadow-black/20">
      <div className="px-4 py-2.5 border-b border-slate-800/80 flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-400 tracking-wide uppercase">
          Change Comparison
        </span>
        <div className="flex items-center gap-3 text-[11px] text-slate-500">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-slate-600" /> Before
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-cyan-400" /> After
          </span>
        </div>
      </div>
      <ReactCompareSlider
        itemOne={<ReactCompareSliderImage src={beforeUrl} alt="Before" />}
        itemTwo={<ReactCompareSliderImage src={afterUrl} alt="After" />}
        style={{ height: "220px" }}
        handle={
          <div className="w-1 h-full bg-cyan-400 shadow-lg shadow-cyan-500/50" />
        }
      />
    </div>
  );
}
