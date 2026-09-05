import { useState } from "react";
import Header from "./components/Header";
import ChatBox from "./components/ChatBox";
import MapOverlay from "./components/MapOverlay";
import ChangeCompare from "./components/ChangeCompare";
import ImageAnnotationView from "./components/ImageAnnotationView";

export default function App() {
  const [result, setResult] = useState(null);
  const [demoMode, setDemoMode] = useState(false);

  const isChangeTask = result?.execution_summary?.task === "change_vqa";
  const [beforeUrl, afterUrl] = result?.previewUrls ?? [];
  const isGeoreferenced = result?.image_georeferenced === true;

  return (
    <div className="h-screen flex flex-col bg-gradient-to-b from-slate-950 to-slate-900">
      <Header demoMode={demoMode} onDemoModeChange={setDemoMode} />
      <main className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-3 p-3 min-h-0 overflow-y-auto">
        <ChatBox onResult={setResult} demoMode={demoMode} />
        <div className="flex flex-col gap-3 min-h-0">
          {isChangeTask && (
            <ChangeCompare beforeUrl={beforeUrl} afterUrl={afterUrl} />
          )}
          <div className="flex-1 min-h-0">
            {result && !isGeoreferenced ? (
              <ImageAnnotationView
                imageUrl={afterUrl || beforeUrl}
                features={result?.geojson?.features}
              />
            ) : (
              <MapOverlay result={result} />
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
