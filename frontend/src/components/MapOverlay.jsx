import {
  MapContainer,
  TileLayer,
  GeoJSON,
  CircleMarker,
  useMap,
} from "react-leaflet";
import { useEffect, useMemo, useState } from "react";
import L from "leaflet";
import EmptyState from "./EmptyState";
import TaskBadge from "./TaskBadge";
import ConfidenceBar from "./ConfidenceBar";

const SOURCE_STYLE = {
  optical: { color: "#22d3ee", fillColor: "#22d3ee" }, // cyan
  sar: { color: "#f97316", fillColor: "#f97316" }, // orange
  default: { color: "#22d3ee", fillColor: "#22d3ee" },
};

function FitToData({ geojson }) {
  const map = useMap();
  useEffect(() => {
    if (!geojson) return;
    const layer = L.geoJSON(geojson);
    const bounds = layer.getBounds();
    if (bounds.isValid()) map.fitBounds(bounds, { padding: [30, 30] });
  }, [geojson, map]);
  return null;
}

function popupHtml(props) {
  const label = props?.label ?? "Region";
  const conf = props?.confidence;
  const source = props?.source;
  return `<div style="font-family:sans-serif;font-size:13px;line-height:1.4">
    <strong>${label}</strong>
    ${source ? `<br/><span style="opacity:.7">source: ${source}</span>` : ""}
    ${conf !== undefined ? `<br/>confidence: ${(conf * 100).toFixed(0)}%` : ""}
  </div>`;
}

export default function MapOverlay({ result }) {
  const geojson = result?.geojson;
  const task = result?.execution_summary?.task;

  // Split features by source (optical / sar) for toggleable layers.
  // Falls back to a single "default" group when no `source` property exists.
  const featureGroups = useMemo(() => {
    if (!geojson?.features?.length) return {};
    const groups = {};
    geojson.features.forEach((f) => {
      const key = f.properties?.source ?? "default";
      groups[key] = groups[key] || [];
      groups[key].push(f);
    });
    return groups;
  }, [geojson]);

  const groupKeys = Object.keys(featureGroups);
  const [visibleLayers, setVisibleLayers] = useState(new Set(groupKeys));

  useEffect(() => {
    setVisibleLayers(new Set(groupKeys));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geojson]);

  const toggleLayer = (key) => {
    setVisibleLayers((prev) => {
      const next = new Set(prev);
      next.has(key) ? next.delete(key) : next.add(key);
      return next;
    });
  };

  return (
    <div className="h-full rounded-2xl border border-slate-800/80 overflow-hidden relative bg-slate-900/60 shadow-xl shadow-black/20">
      {!geojson ? (
        <EmptyState />
      ) : (
        <>
          <MapContainer
            center={[20, 78]}
            zoom={4}
            className="h-full w-full"
            zoomControl={false}
          >
            <TileLayer
              attribution="&copy; OpenStreetMap contributors"
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />

            {groupKeys.map((key) => {
              if (!visibleLayers.has(key)) return null;
              const style = SOURCE_STYLE[key] ?? SOURCE_STYLE.default;
              const features = featureGroups[key];

              return features.map((feature, i) => {
                if (feature.geometry?.type === "Point") {
                  const [lon, lat] = feature.geometry.coordinates;
                  return (
                    <CircleMarker
                      key={`${key}-${i}`}
                      center={[lat, lon]}
                      radius={8}
                      pathOptions={{
                        color: style.color,
                        fillColor: style.fillColor,
                        fillOpacity: 0.6,
                        weight: 2,
                      }}
                      eventHandlers={{
                        add: (e) =>
                          e.target.bindPopup(popupHtml(feature.properties)),
                      }}
                    />
                  );
                }
                return (
                  <GeoJSON
                    key={`${key}-${i}`}
                    data={feature}
                    style={{
                      color: style.color,
                      weight: 2,
                      fillColor: style.fillColor,
                      fillOpacity: 0.15,
                    }}
                    onEachFeature={(f, layer) =>
                      layer.bindPopup(popupHtml(f.properties))
                    }
                  />
                );
              });
            })}

            <FitToData geojson={geojson} />
          </MapContainer>

          {/* Layer toggle chips — only shown when there's more than one source (e.g. optical vs SAR) */}
          {groupKeys.length > 1 && (
            <div className="absolute top-3 left-3 flex gap-1.5 z-[500]">
              {groupKeys.map((key) => {
                const style = SOURCE_STYLE[key] ?? SOURCE_STYLE.default;
                const active = visibleLayers.has(key);
                return (
                  <button
                    key={key}
                    onClick={() => toggleLayer(key)}
                    className={`flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-full border backdrop-blur transition-all ${
                      active
                        ? "bg-slate-950/80 border-slate-700 text-slate-200"
                        : "bg-slate-950/40 border-slate-800 text-slate-600"
                    }`}
                  >
                    <span
                      className="w-2 h-2 rounded-full"
                      style={{
                        backgroundColor: active ? style.color : "#475569",
                      }}
                    />
                    {key === "default" ? "Result" : key.toUpperCase()}
                  </button>
                );
              })}
            </div>
          )}

          {/* Result card */}
          {result?.text && (
            <div className="absolute bottom-3 left-3 right-3 bg-slate-950/90 backdrop-blur-md border border-slate-800 rounded-xl px-4 py-3 shadow-2xl shadow-black/40 z-[500]">
              <div className="flex items-start justify-between gap-3 mb-1.5">
                {task && <TaskBadge task={task} />}
                <ConfidenceBar value={result.confidence} />
              </div>
              <p className="text-sm text-slate-200 leading-relaxed">
                {result.text}
              </p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
