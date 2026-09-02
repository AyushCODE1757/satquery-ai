import DemoModeToggle from "./DemoModeToggle";

export default function Header({ demoMode, onDemoModeChange }) {
  return (
    <header className="flex items-center justify-between px-5 py-3 border-b border-slate-800/80 bg-slate-950/90 backdrop-blur sticky top-0 z-20">
      <div className="flex items-center gap-2.5">
        <div className="h-8 w-8 rounded-lg bg-gradient-to-br from-cyan-400 to-blue-600 flex items-center justify-center font-bold text-slate-950 shadow-lg shadow-cyan-500/20">
          S
        </div>
        <div>
          <h1 className="text-slate-100 font-semibold leading-none tracking-tight">
            SatQuery AI
          </h1>
          <p className="text-slate-500 text-[11px] mt-0.5">
            Multimodal Remote Sensing Assistant
          </p>
        </div>
      </div>
      <div className="flex items-center gap-3">
        <DemoModeToggle enabled={demoMode} onChange={onDemoModeChange} />
        <span className="text-[11px] px-2.5 py-1 rounded-full border border-cyan-800/60 text-cyan-400 bg-cyan-950/30">
          Team F6
        </span>
      </div>
    </header>
  );
}
