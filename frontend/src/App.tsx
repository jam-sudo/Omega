import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/layout/Sidebar";
import StatusBar from "./components/layout/StatusBar";
import Dashboard from "./pages/Dashboard";
import Predict from "./pages/Predict";
import Compare from "./pages/Compare";
import DoseOptimizer from "./pages/DoseOptimizer";
import DDIChecker from "./pages/DDIChecker";
import PopulationPK from "./pages/PopulationPK";
import Reports from "./pages/Reports";

export default function App() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/dose-optimize" element={<DoseOptimizer />} />
            <Route path="/ddi" element={<DDIChecker />} />
            <Route path="/population" element={<PopulationPK />} />
            <Route path="/reports" element={<Reports />} />
          </Routes>
        </main>
        <StatusBar />
      </div>
    </div>
  );
}
