export default function EmptyState() {
  return (
    <div className="h-full flex flex-col items-center justify-center text-center px-6 text-slate-500">
      <div className="h-14 w-14 rounded-full border border-slate-700 flex items-center justify-center mb-3 text-2xl">
        🛰️
      </div>
      <p className="text-slate-300 font-medium">No results yet</p>
      <p className="text-sm mt-1 max-w-xs">
        Upload an image (or pair) and ask a question — evidence and bounding
        boxes will render here.
      </p>
    </div>
  );
}
