import { useState } from "react";
import Header from "./components/Header";
import ChatBox from "./components/ChatBox";
import MapOverlay from "./components/MapOverlay";
import mock from "./mocks/mockFinal.json";

export default function App() {
  const [result, setResult] = useState(mock);
  const [demoMode, setDemoMode] = useState(false); // default false — real inference, per AK's guidance

  return (
    <div className="h-screen flex flex-col bg-gradient-to-b from-slate-950 to-slate-900">
      <Header demoMode={demoMode} onDemoModeChange={setDemoMode} />
      <main className="flex-1 grid grid-cols-1 md:grid-cols-2 gap-3 p-3 min-h-0">
        <ChatBox onResult={setResult} demoMode={demoMode} />
        <MapOverlay result={result} />
      </main>
    </div>
  );
}
