export default function ImageAnnotationView({ imageUrl, features = [] }) {
  if (!imageUrl) return null;

  return (
    <div className="relative rounded-2xl border border-slate-800/80 overflow-hidden bg-slate-900/60 shadow-xl shadow-black/20">
      <div className="relative w-full">
        <img
          src={imageUrl}
          alt="Uploaded satellite imagery"
          className="w-full h-auto block"
        />
        <svg
          className="absolute inset-0 w-full h-full"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
        >
          {features.map((f, i) => {
            const box = f.properties?.image_space_bbox;
            if (!box) return null;
            const [x1, y1, x2, y2] = box;
            return (
              <rect
                key={i}
                x={x1 * 100}
                y={y1 * 100}
                width={(x2 - x1) * 100}
                height={(y2 - y1) * 100}
                fill="rgba(34, 211, 238, 0.15)"
                stroke="#22d3ee"
                strokeWidth="0.5"
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
        </svg>
      </div>
      <div className="absolute top-2 left-2 text-[11px] px-2 py-0.5 rounded-full bg-slate-950/80 border border-slate-700 text-slate-300">
        Image view (not georeferenced — showing detection on the actual photo)
      </div>
    </div>
  );
}
