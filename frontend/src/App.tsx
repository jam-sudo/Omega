import { lazy, Suspense } from "react";
import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/layout/Sidebar";
import StatusBar from "./components/layout/StatusBar";

const Dashboard = lazy(() => import("./pages/Dashboard"));
const Predict = lazy(() => import("./pages/Predict"));
const Compare = lazy(() => import("./pages/Compare"));
const DoseOptimizer = lazy(() => import("./pages/DoseOptimizer"));
const DDIChecker = lazy(() => import("./pages/DDIChecker"));
const PopulationPK = lazy(() => import("./pages/PopulationPK"));
const Reports = lazy(() => import("./pages/Reports"));

export default function App() {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col overflow-hidden">
        <main className="flex-1 overflow-auto">
          <Suspense
            fallback={
              <div className="flex h-full items-center justify-center text-sm text-[var(--text-muted)]">
                Loading...
              </div>
            }
          >
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predict" element={<Predict />} />
              <Route path="/compare" element={<Compare />} />
              <Route path="/dose-optimize" element={<DoseOptimizer />} />
              <Route path="/ddi" element={<DDIChecker />} />
              <Route path="/population" element={<PopulationPK />} />
              <Route path="/reports" element={<Reports />} />
            </Routes>
          </Suspense>
        </main>
        <StatusBar />
      </div>
    </div>
  );
}
