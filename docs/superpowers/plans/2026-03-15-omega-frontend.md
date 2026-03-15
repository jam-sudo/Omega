# Omega PBPK Frontend Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a React SPA frontend for the Omega PBPK pharmacokinetic prediction platform, connecting to the existing FastAPI backend.

**Architecture:** React 18 + Vite + TypeScript SPA with 7 pages (Dashboard, Predict, Compare, Dose Optimizer, DDI, PopPK, Reports). Collapsible sidebar layout. TanStack Query for API state. Dark mode with shadcn/ui components. All API calls go through a Vite proxy to the existing FastAPI backend (`omega serve`).

**Tech Stack:** React 18, Vite, TypeScript, Tailwind CSS, shadcn/ui, Recharts, TanStack Query, smiles-drawer, react-router-dom

**Spec:** `docs/superpowers/specs/2026-03-15-omega-frontend-design.md`

---

## Chunk 1: Project Scaffold + Layout Shell

### Task 1: Initialize Vite + React + TypeScript project

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/index.html`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tailwind.config.ts`
- Create: `frontend/postcss.config.js`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/index.css`

- [ ] **Step 1: Scaffold Vite project**

```bash
cd /home/jam/Omega
npm create vite@latest frontend -- --template react-ts
cd frontend
```

- [ ] **Step 2: Install core dependencies**

```bash
npm install react-router-dom @tanstack/react-query recharts smiles-drawer
npm install -D tailwindcss @tailwindcss/vite
```

- [ ] **Step 3: Configure Vite with API proxy + Tailwind**

Replace `frontend/vite.config.ts`:
```typescript
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
```

- [ ] **Step 4: Set up Tailwind with dark theme**

Replace `frontend/src/index.css`:
```css
@import "tailwindcss";

:root {
  --background: #0a0a0a;
  --surface: #141414;
  --border: #262626;
  --text: #fafafa;
  --text-muted: #a1a1aa;
  --accent: #3b82f6;
  --success: #22c55e;
  --warning: #eab308;
  --danger: #ef4444;
}

body {
  background: var(--background);
  color: var(--text);
  font-family: "Inter", system-ui, -apple-system, sans-serif;
  margin: 0;
}

* {
  border-color: var(--border);
}

.tabular-nums {
  font-variant-numeric: tabular-nums;
}
```

- [ ] **Step 5: Set up entry point**

Replace `frontend/src/main.tsx`:
```tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, staleTime: 5 * 60 * 1000 },
  },
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
);
```

- [ ] **Step 6: Verify dev server starts**

```bash
cd /home/jam/Omega/frontend
npm run dev
```

Open `http://localhost:5173` — should show default Vite React page.

- [ ] **Step 7: Commit**

```bash
cd /home/jam/Omega
git add frontend/
git commit -m "feat(frontend): scaffold Vite + React + TS + Tailwind project"
```

---

### Task 2: Sidebar + Layout Shell

**Files:**
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/components/layout/Sidebar.tsx`
- Create: `frontend/src/components/layout/PageShell.tsx`
- Create: `frontend/src/components/layout/StatusBar.tsx`
- Create: `frontend/src/pages/Dashboard.tsx` (placeholder)
- Create: `frontend/src/pages/Predict.tsx` (placeholder)
- Create: `frontend/src/pages/Compare.tsx` (placeholder)
- Create: `frontend/src/pages/DoseOptimizer.tsx` (placeholder)
- Create: `frontend/src/pages/DDIChecker.tsx` (placeholder)
- Create: `frontend/src/pages/PopulationPK.tsx` (placeholder)
- Create: `frontend/src/pages/Reports.tsx` (placeholder)

- [ ] **Step 1: Create Sidebar component**

Create `frontend/src/components/layout/Sidebar.tsx`:
```tsx
import { NavLink } from "react-router-dom";
import { useState } from "react";

const NAV_ITEMS = [
  { path: "/", label: "Dashboard", icon: "🏠" },
  { path: "/predict", label: "Predict", icon: "🔬" },
  { path: "/compare", label: "Compare", icon: "⚖️" },
  { path: "/dose-optimize", label: "Dose Opt", icon: "💊" },
  { path: "/ddi", label: "DDI", icon: "⚠️" },
  { path: "/population", label: "PopPK", icon: "👥" },
  { path: "/reports", label: "Reports", icon: "📄" },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={`h-screen sticky top-0 flex flex-col border-r transition-all duration-200 ${
        collapsed ? "w-[60px]" : "w-[240px]"
      }`}
      style={{ background: "var(--surface)" }}
    >
      <div className="p-4 font-bold text-lg flex items-center gap-2">
        <span className="text-xl">Ω</span>
        {!collapsed && <span>Omega PBPK</span>}
      </div>

      <nav className="flex-1 flex flex-col gap-1 px-2">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${
                isActive
                  ? "bg-blue-500/10 text-blue-400"
                  : "text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-white/5"
              }`
            }
          >
            <span>{item.icon}</span>
            {!collapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      <button
        onClick={() => setCollapsed(!collapsed)}
        className="p-3 text-[var(--text-muted)] hover:text-[var(--text)] text-xs"
      >
        {collapsed ? "→" : "← Collapse"}
      </button>
    </aside>
  );
}
```

- [ ] **Step 2: Create PageShell component**

Create `frontend/src/components/layout/PageShell.tsx`:
```tsx
import { useEffect } from "react";

interface Props {
  title: string;
  children: React.ReactNode;
}

export default function PageShell({ title, children }: Props) {
  useEffect(() => {
    document.title = `${title} — Omega PBPK`;
  }, [title]);

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <h1 className="text-2xl font-semibold mb-6">{title}</h1>
      {children}
    </div>
  );
}
```

- [ ] **Step 3: Create StatusBar component**

Create `frontend/src/components/layout/StatusBar.tsx`:
```tsx
import { useQuery } from "@tanstack/react-query";

export default function StatusBar() {
  const { data, isError } = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const start = performance.now();
      const res = await fetch("/api/health");
      const latency = Math.round(performance.now() - start);
      if (!res.ok) throw new Error("API unreachable");
      return { connected: true, latency };
    },
    refetchInterval: 30_000,
  });

  const connected = data?.connected && !isError;

  return (
    <footer
      className="h-8 border-t flex items-center px-4 text-xs gap-4"
      style={{ background: "var(--surface)", color: "var(--text-muted)" }}
    >
      <span>
        API:{" "}
        <span className={connected ? "text-green-400" : "text-red-400"}>●</span>{" "}
        {connected ? "connected" : "disconnected"}
      </span>
      {data?.latency && <span>Last: {data.latency}ms</span>}
      <span className="ml-auto">Omega v0.1</span>
    </footer>
  );
}
```

- [ ] **Step 4: Create placeholder pages**

Create 7 placeholder pages in `frontend/src/pages/`. Each follows this pattern:

`frontend/src/pages/Dashboard.tsx`:
```tsx
import PageShell from "../components/layout/PageShell";
export default function Dashboard() {
  return <PageShell title="Dashboard"><p className="text-[var(--text-muted)]">Coming soon.</p></PageShell>;
}
```

Repeat for: `Predict.tsx`, `Compare.tsx`, `DoseOptimizer.tsx`, `DDIChecker.tsx`, `PopulationPK.tsx`, `Reports.tsx` (changing title for each).

- [ ] **Step 5: Wire up App.tsx with router + layout**

Create `frontend/src/App.tsx`:
```tsx
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
```

- [ ] **Step 6: Verify — all routes render with sidebar**

```bash
cd /home/jam/Omega/frontend && npm run dev
```

Navigate to each route. Sidebar should highlight active page. StatusBar shows at bottom.

- [ ] **Step 7: Commit**

```bash
git add frontend/
git commit -m "feat(frontend): sidebar layout + routing + status bar"
```

---

### Task 3: API Client + Types + History Utils

**Files:**
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/api/orchestrate.ts`
- Create: `frontend/src/api/types.ts`
- Create: `frontend/src/lib/utils.ts`
- Create: `frontend/src/lib/history.ts`
- Create: `frontend/src/components/ErrorAlert.tsx`

- [ ] **Step 1: Create API types (manual for now, auto-generate later)**

Create `frontend/src/api/types.ts`:
```typescript
// Manual types matching FastAPI models. Replace with openapi-typescript output later.

export interface PredictRequest {
  smiles: string;
  dose_mg?: number;
  route?: string;
  duration_h?: number;
}

export interface FullPredictResponse {
  drug_name: string;
  smiles: string;
  cmax_mg_L: number;
  tmax_h: number;
  auc0t_mg_h_L: number;
  t_half_h: number;
  time_h: number[];
  cp_mg_L: number[];
  confidence: string;
  warnings: string[];
  adme: Record<string, number | string>;
  // Percentiles (from /predict/full)
  cmax_p5?: number;
  cmax_p50?: number;
  cmax_p95?: number;
  auc_p5?: number;
  auc_p50?: number;
  auc_p95?: number;
  risk_flags?: Record<string, boolean>;
  overall_risk_level?: string;
}

export interface DrugRequest {
  name: string;
  mw: number;
  logP: number;
  fup: number;
  rbp: number;
  clint_hepatic_L_per_h: number;
  peff: number;
  dose_mg: number;
}

export interface DDISimulateRequest {
  victim_drug: DrugRequest;
  perpetrator_name: string;
  perpetrator_ki_uM: number;
  perpetrator_target_enzyme: string;
  perpetrator_mechanism?: string;
  perpetrator_cmax_uM?: number;
}

export interface DDIResponse {
  auc_ratio: number;
  cmax_ratio: number;
  interaction_magnitude: string;
  time_h: number[];
  cp_alone: number[];
  cp_with_inhibitor: number[];
}

export interface DoseOptimizeRequest {
  drug: DrugRequest;
  cmin_mg_L: number;
  cmax_mg_L: number;
  dose_range_mg: [number, number];
}

export interface DoseOptimizeResponse {
  optimal_dose_mg: number;
  css_max: number;
  css_min: number;
  auc_ss: number;
}

export interface PopulationRequest {
  drug: DrugRequest;
  n_subjects: number;
  dose_mg: number;
  route?: string;
}

export interface PopulationResponse {
  n_subjects: number;
  cmax_median: number;
  cmax_mean: number;
  cmax_cv_pct: number;
  cmax_p5: number;
  cmax_p50: number;
  cmax_p95: number;
  auc_median: number;
  auc_mean: number;
  auc_cv_pct: number;
}

export interface HealthResponse {
  status: string;
}
```

- [ ] **Step 2: Create API client**

Create `frontend/src/api/client.ts`:
```typescript
import type {
  PredictRequest,
  FullPredictResponse,
  DDISimulateRequest,
  DDIResponse,
  DoseOptimizeRequest,
  DoseOptimizeResponse,
  PopulationRequest,
  PopulationResponse,
  HealthResponse,
} from "./types";

const BASE = "/api";

async function post<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

export const api = {
  predict: (req: PredictRequest, signal?: AbortSignal) =>
    post<FullPredictResponse>("/predict/full", req, signal),
  ddiSimulate: (req: DDISimulateRequest, signal?: AbortSignal) =>
    post<DDIResponse>("/ddi/simulate", req, signal),
  doseOptimize: (req: DoseOptimizeRequest, signal?: AbortSignal) =>
    post<DoseOptimizeResponse>("/dose/optimize", req, signal),
  population: (req: PopulationRequest, signal?: AbortSignal) =>
    post<PopulationResponse>("/population", req, signal),
  report: (req: unknown, signal?: AbortSignal) =>
    post<{ data: string; encoding: string }>("/report", req, signal),
  health: () => get<HealthResponse>("/health"),
};
```

- [ ] **Step 3: Create orchestration helper**

Create `frontend/src/api/orchestrate.ts`:
```typescript
import type { FullPredictResponse, DrugRequest } from "./types";

export function admeToDrugRequest(
  response: FullPredictResponse,
  dose_mg: number,
): DrugRequest {
  const adme = response.adme;
  return {
    name: response.drug_name || response.smiles.slice(0, 20),
    mw: Number(adme.mw) || 300,
    logP: Number(adme.logP) || 2,
    fup: Number(adme.fup) || 0.1,
    rbp: Number(adme.rbp) || 1.0,
    clint_hepatic_L_per_h: Number(adme.clint_hepatic_L_per_h) || 0.2,
    peff: Number(adme.peff) || 1.0,
    dose_mg,
  };
}
```

- [ ] **Step 4: Create history utils**

Create `frontend/src/lib/history.ts`:
```typescript
const KEY = "omega_history";
const MAX = 20;

export interface HistoryEntry {
  smiles: string;
  dose: number;
  route: string;
  cmax: number;
  auc: number;
  thalf: number;
  timestamp: number;
}

export function getHistory(): HistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) || "[]");
  } catch {
    return [];
  }
}

export function addToHistory(entry: HistoryEntry): void {
  const list = getHistory();
  // Remove duplicate SMILES
  const filtered = list.filter((e) => e.smiles !== entry.smiles);
  filtered.unshift(entry);
  localStorage.setItem(KEY, JSON.stringify(filtered.slice(0, MAX)));
}

export function clearHistory(): void {
  localStorage.removeItem(KEY);
}
```

- [ ] **Step 5: Create formatting utils**

Create `frontend/src/lib/utils.ts`:
```typescript
export function formatNum(n: number, decimals = 2): string {
  if (n === 0) return "0";
  if (Math.abs(n) < 0.001) return n.toExponential(1);
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  return n.toFixed(decimals);
}

export function confidenceColor(level: string): string {
  switch (level) {
    case "high": return "var(--success)";
    case "medium": return "var(--warning)";
    case "low": return "var(--danger)";
    default: return "var(--text-muted)";
  }
}

export function downloadBlob(data: string, filename: string, mime: string): void {
  const blob = new Blob([data], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportCSV(timeH: number[], cpMgL: number[], filename: string): void {
  const header = "time_h,cp_mg_L\n";
  const rows = timeH.map((t, i) => `${t},${cpMgL[i]}`).join("\n");
  downloadBlob(header + rows, filename, "text/csv");
}
```

- [ ] **Step 6: Create ErrorAlert component**

Create `frontend/src/components/ErrorAlert.tsx`:
```tsx
interface Props {
  message: string;
}

export default function ErrorAlert({ message }: Props) {
  return (
    <div className="border border-red-500/30 bg-red-500/10 rounded-md px-4 py-3 text-sm text-red-400 flex items-center gap-2 mt-4">
      <span>⚠</span>
      <span>{message}</span>
    </div>
  );
}
```

- [ ] **Step 7: Commit**

```bash
git add frontend/src/api/ frontend/src/lib/ frontend/src/components/ErrorAlert.tsx
git commit -m "feat(frontend): API client, types, orchestration, history utils"
```

---

## Chunk 2: Shared Components + Predict Page

### Task 4: SmilesInput + molecule preview

**Files:**
- Create: `frontend/src/components/SmilesInput.tsx`

- [ ] **Step 1: Create SmilesInput with smiles-drawer preview**

Create `frontend/src/components/SmilesInput.tsx`:
```tsx
import { useEffect, useRef } from "react";

interface Props {
  value: string;
  onChange: (v: string) => void;
  showPreview?: boolean;
}

export default function SmilesInput({ value, onChange, showPreview = true }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!showPreview || !value || !canvasRef.current) return;

    let cancelled = false;
    import("smiles-drawer").then((mod) => {
      if (cancelled) return;
      const SmilesDrawer = mod.default || mod;
      const drawer = new SmilesDrawer.Drawer({ width: 200, height: 150, themes: { dark: { C: "#fafafa", O: "#ef4444", N: "#3b82f6", S: "#eab308", H: "#a1a1aa", BACKGROUND: "#141414" } } });
      try {
        SmilesDrawer.parse(value, (tree: unknown) => {
          if (!cancelled) drawer.draw(tree, canvasRef.current!, "dark");
        });
      } catch {
        // Invalid SMILES — clear canvas
        const ctx = canvasRef.current?.getContext("2d");
        if (ctx) ctx.clearRect(0, 0, 200, 150);
      }
    });

    return () => { cancelled = true; };
  }, [value, showPreview]);

  return (
    <div className="flex gap-4 items-start">
      <div className="flex-1">
        <label className="block text-xs font-medium text-[var(--text-muted)] mb-1 uppercase tracking-wide">
          SMILES
        </label>
        <input
          type="text"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="e.g. Cn1c(=O)c2c(ncn2C)n(C)c1=O"
          className="w-full px-3 py-2 rounded-md border bg-[var(--surface)] text-[var(--text)] text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
        />
      </div>
      {showPreview && (
        <canvas
          ref={canvasRef}
          width={200}
          height={150}
          className="rounded-md border"
          style={{ background: "var(--surface)" }}
        />
      )}
    </div>
  );
}
```

- [ ] **Step 2: Verify — import SmilesInput in Predict page placeholder, type a SMILES, see molecule**

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/SmilesInput.tsx
git commit -m "feat(frontend): SmilesInput with smiles-drawer molecule preview"
```

---

### Task 5: PKCards + PKCurve + ADMETable + WarningBadge

**Files:**
- Create: `frontend/src/components/charts/PKCards.tsx`
- Create: `frontend/src/components/charts/PKCurve.tsx`
- Create: `frontend/src/components/ADMETable.tsx`
- Create: `frontend/src/components/WarningBadge.tsx`

- [ ] **Step 1: Create PKCards**

Create `frontend/src/components/charts/PKCards.tsx`:
```tsx
import { formatNum, confidenceColor } from "../../lib/utils";

interface CardData {
  label: string;
  value: number;
  unit: string;
  p5?: number;
  p95?: number;
}

interface Props {
  cards: CardData[];
  confidence?: string;
}

export default function PKCards({ cards, confidence }: Props) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {cards.map((c) => (
        <div key={c.label} className="border rounded-lg p-4" style={{ background: "var(--surface)" }}>
          <div className="text-xs text-[var(--text-muted)] uppercase tracking-wide mb-1">
            {c.label}
          </div>
          <div className="text-2xl font-semibold tabular-nums">{formatNum(c.value)}</div>
          <div className="text-xs text-[var(--text-muted)]">{c.unit}</div>
          {c.p5 !== undefined && c.p95 !== undefined && (
            <div className="text-xs text-[var(--text-muted)] mt-1">
              CI: {formatNum(c.p5)} — {formatNum(c.p95)}
            </div>
          )}
        </div>
      ))}
      {confidence && (
        <div className="col-span-full text-xs flex items-center gap-2">
          <span
            className="inline-block w-2 h-2 rounded-full"
            style={{ background: confidenceColor(confidence) }}
          />
          Confidence: {confidence}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Create PKCurve**

Create `frontend/src/components/charts/PKCurve.tsx`:
```tsx
import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from "recharts";
import { exportCSV } from "../../lib/utils";

interface Dataset {
  time: number[];
  conc: number[];
  label: string;
  color: string;
}

interface Props {
  datasets: Dataset[];
  mecLine?: number;
  mtcLine?: number;
}

const COLORS = ["#3b82f6", "#f97316", "#22c55e", "#a855f7"];

export default function PKCurve({ datasets, mecLine, mtcLine }: Props) {
  const [logScale, setLogScale] = useState(false);

  // Merge datasets into chart data
  const maxLen = Math.max(...datasets.map((d) => d.time.length));
  const chartData = Array.from({ length: maxLen }, (_, i) => {
    const point: Record<string, number> = { time: datasets[0]?.time[i] ?? i * 0.1 };
    datasets.forEach((ds, idx) => {
      let val = ds.conc[i] ?? 0;
      if (logScale) val = val > 0 ? val : 1e-6;
      point[`conc_${idx}`] = val;
    });
    return point;
  });

  const handleExportCSV = () => {
    if (datasets[0]) exportCSV(datasets[0].time, datasets[0].conc, "pk_curve.csv");
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium">Concentration-Time Profile</span>
        <div className="flex gap-2 items-center">
          <button
            onClick={() => setLogScale(false)}
            className={`px-2 py-1 text-xs rounded ${!logScale ? "bg-blue-500/20 text-blue-400" : "text-[var(--text-muted)]"}`}
          >
            Linear
          </button>
          <button
            onClick={() => setLogScale(true)}
            className={`px-2 py-1 text-xs rounded ${logScale ? "bg-blue-500/20 text-blue-400" : "text-[var(--text-muted)]"}`}
          >
            Log
          </button>
        </div>
      </div>

      <div className="border rounded-lg p-4" style={{ background: "var(--surface)" }}>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis dataKey="time" stroke="var(--text-muted)" tick={{ fontSize: 11 }} label={{ value: "Time (h)", position: "bottom", offset: -5, style: { fill: "var(--text-muted)", fontSize: 11 } }} />
            <YAxis
              scale={logScale ? "log" : "linear"}
              domain={logScale ? ["auto", "auto"] : [0, "auto"]}
              stroke="var(--text-muted)"
              tick={{ fontSize: 11 }}
              label={{ value: "Conc (mg/L)", angle: -90, position: "insideLeft", style: { fill: "var(--text-muted)", fontSize: 11 } }}
            />
            <Tooltip
              contentStyle={{ background: "var(--surface)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12 }}
              labelFormatter={(v) => `${Number(v).toFixed(1)}h`}
            />
            {datasets.map((ds, idx) => (
              <Line
                key={ds.label}
                type="monotone"
                dataKey={`conc_${idx}`}
                name={ds.label}
                stroke={ds.color || COLORS[idx]}
                strokeWidth={2}
                dot={false}
                animationDuration={800}
              />
            ))}
            {mecLine !== undefined && (
              <ReferenceLine y={mecLine} stroke="var(--success)" strokeDasharray="5 5" label={{ value: "MEC", fill: "var(--success)", fontSize: 10 }} />
            )}
            {mtcLine !== undefined && (
              <ReferenceLine y={mtcLine} stroke="var(--danger)" strokeDasharray="5 5" label={{ value: "MTC", fill: "var(--danger)", fontSize: 10 }} />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="flex gap-2 mt-2">
        <button onClick={() => navigator.clipboard.writeText(`Cmax\tAUC\nt½`)} className="text-xs px-2 py-1 rounded border text-[var(--text-muted)] hover:text-[var(--text)]">📋 Copy</button>
        <button onClick={handleExportCSV} className="text-xs px-2 py-1 rounded border text-[var(--text-muted)] hover:text-[var(--text)]">📥 CSV</button>
      </div>

      {datasets.length > 1 && (
        <div className="flex gap-4 mt-2">
          {datasets.map((ds, idx) => (
            <span key={ds.label} className="text-xs flex items-center gap-1">
              <span className="w-3 h-0.5 inline-block" style={{ background: ds.color || COLORS[idx] }} />
              {ds.label}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Create ADMETable**

Create `frontend/src/components/ADMETable.tsx`:
```tsx
import { formatNum } from "../lib/utils";

interface Props {
  adme: Record<string, number | string>;
}

const DISPLAY_PROPS = [
  { key: "mw", label: "MW", unit: "g/mol" },
  { key: "logP", label: "logP", unit: "" },
  { key: "logS", label: "logS", unit: "" },
  { key: "fup", label: "fup", unit: "" },
  { key: "rbp", label: "RBP", unit: "" },
  { key: "peff", label: "Peff", unit: "×10⁻⁴ cm/s" },
  { key: "clint_3a4", label: "CLint 3A4", unit: "µL/min/pmol" },
  { key: "herg_ic50_uM", label: "hERG IC50", unit: "µM" },
];

export default function ADMETable({ adme }: Props) {
  return (
    <div className="border rounded-lg overflow-hidden" style={{ background: "var(--surface)" }}>
      <div className="px-4 py-2 text-xs font-medium uppercase tracking-wide text-[var(--text-muted)] border-b">
        ADME Properties
      </div>
      <table className="w-full text-sm">
        <tbody>
          {DISPLAY_PROPS.map((prop) => {
            const val = adme[prop.key];
            if (val === undefined) return null;
            const lo = adme[`${prop.key}_lo`];
            const hi = adme[`${prop.key}_hi`];
            return (
              <tr key={prop.key} className="border-b last:border-0 hover:bg-white/5">
                <td className="px-4 py-2 text-[var(--text-muted)]">{prop.label}</td>
                <td className="px-4 py-2 tabular-nums font-medium">
                  {typeof val === "number" ? formatNum(val) : val}
                  {prop.unit && <span className="text-[var(--text-muted)] ml-1 text-xs">{prop.unit}</span>}
                </td>
                <td className="px-4 py-2 text-xs text-[var(--text-muted)]">
                  {lo !== undefined && hi !== undefined && (
                    <span>[{formatNum(Number(lo))} — {formatNum(Number(hi))}]</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
```

- [ ] **Step 4: Create WarningBadge**

Create `frontend/src/components/WarningBadge.tsx`:
```tsx
interface Props {
  warnings: string[];
  riskLevel?: string;
}

export default function WarningBadge({ warnings, riskLevel }: Props) {
  if (warnings.length === 0 && !riskLevel) return null;

  return (
    <div className="border rounded-lg p-4" style={{ background: "var(--surface)" }}>
      <div className="text-xs font-medium uppercase tracking-wide text-[var(--text-muted)] mb-2">
        Warnings & Risk
      </div>
      {warnings.map((w, i) => (
        <div key={i} className="text-sm text-yellow-400 flex items-center gap-2 mb-1">
          <span>⚠</span> {w}
        </div>
      ))}
      {riskLevel && (
        <div className="text-xs text-[var(--text-muted)] mt-2">
          Overall risk: <span className="font-medium text-[var(--text)]">{riskLevel}</span>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/
git commit -m "feat(frontend): PKCards, PKCurve, ADMETable, WarningBadge components"
```

---

### Task 6: Predict Page (full implementation)

**Files:**
- Modify: `frontend/src/pages/Predict.tsx`

- [ ] **Step 1: Implement Predict page**

Replace `frontend/src/pages/Predict.tsx`:
```tsx
import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import PageShell from "../components/layout/PageShell";
import SmilesInput from "../components/SmilesInput";
import PKCards from "../components/charts/PKCards";
import PKCurve from "../components/charts/PKCurve";
import ADMETable from "../components/ADMETable";
import WarningBadge from "../components/WarningBadge";
import ErrorAlert from "../components/ErrorAlert";
import { api } from "../api/client";
import { addToHistory } from "../lib/history";

export default function Predict() {
  const [params] = useSearchParams();
  const [smiles, setSmiles] = useState(params.get("smiles") || "");
  const [dose, setDose] = useState(Number(params.get("dose")) || 100);
  const [route, setRoute] = useState(params.get("route") || "oral");

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ["predict", smiles, dose, route],
    queryFn: ({ signal }) => api.predict({ smiles, dose_mg: dose, route }, signal),
    enabled: false,
  });

  // Save to history when prediction succeeds
  useEffect(() => {
    if (data) {
      addToHistory({
        smiles: data.smiles,
        dose, route,
        cmax: data.cmax_mg_L,
        auc: data.auc0t_mg_h_L,
        thalf: data.t_half_h,
        timestamp: Date.now(),
      });
    }
  }, [data]);

  const handleSubmit = () => {
    if (smiles.trim()) refetch();
  };

  // Ctrl+Enter shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.ctrlKey && e.key === "Enter") handleSubmit();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [smiles, dose, route]);

  return (
    <PageShell title="Predict">
      {/* Input Section */}
      <div className="flex gap-4 items-end mb-6">
        <div className="flex-1">
          <SmilesInput value={smiles} onChange={setSmiles} />
        </div>
        <div>
          <label className="block text-xs font-medium text-[var(--text-muted)] mb-1 uppercase tracking-wide">Dose (mg)</label>
          <input type="number" value={dose} onChange={(e) => setDose(Number(e.target.value))} min={0.1} step={0.1}
            className="w-24 px-3 py-2 rounded-md border bg-[var(--surface)] text-[var(--text)] text-sm" />
        </div>
        <div>
          <label className="block text-xs font-medium text-[var(--text-muted)] mb-1 uppercase tracking-wide">Route</label>
          <select value={route} onChange={(e) => setRoute(e.target.value)}
            className="px-3 py-2 rounded-md border bg-[var(--surface)] text-[var(--text)] text-sm">
            <option value="oral">Oral</option>
            <option value="iv">IV</option>
          </select>
        </div>
        <button onClick={handleSubmit} disabled={isLoading || !smiles.trim()}
          className="px-6 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed">
          {isLoading ? "Running..." : "Predict"}
        </button>
      </div>

      {error && <ErrorAlert message={(error as Error).message} />}

      {/* Skeleton loading */}
      {isLoading && (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-4">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-24 rounded-lg animate-pulse" style={{ background: "var(--surface)" }} />
            ))}
          </div>
          <div className="h-80 rounded-lg animate-pulse" style={{ background: "var(--surface)" }} />
        </div>
      )}

      {/* Results */}
      {data && !isLoading && (
        <div className="space-y-6">
          <PKCards
            cards={[
              { label: "Cmax", value: data.cmax_mg_L, unit: "mg/L", p5: data.cmax_p5, p95: data.cmax_p95 },
              { label: "Tmax", value: data.tmax_h, unit: "h" },
              { label: "AUC₀₋ₜ", value: data.auc0t_mg_h_L, unit: "mg·h/L", p5: data.auc_p5, p95: data.auc_p95 },
              { label: "t½", value: data.t_half_h, unit: "h" },
            ]}
            confidence={data.confidence}
          />

          <PKCurve
            datasets={[{
              time: data.time_h,
              conc: data.cp_mg_L,
              label: data.drug_name || smiles.slice(0, 15),
              color: "#3b82f6",
            }]}
          />

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ADMETable adme={data.adme} />
            <WarningBadge warnings={data.warnings} riskLevel={data.overall_risk_level} />
          </div>
        </div>
      )}
    </PageShell>
  );
}
```

- [ ] **Step 2: Start FastAPI backend and verify**

```bash
# Terminal 1:
cd /home/jam/Omega && source .venv/bin/activate && omega serve start

# Terminal 2:
cd /home/jam/Omega/frontend && npm run dev
```

Navigate to `http://localhost:5173/predict`. Enter caffeine SMILES `Cn1c(=O)c2c(ncn2C)n(C)c1=O`, dose 200, click Predict. Should see PK cards, C(t) curve, ADME table.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/Predict.tsx
git commit -m "feat(frontend): full Predict page with PK cards, chart, ADME table"
```

---

## Chunk 3: Dashboard + Compare + dev.sh

### Task 7: Dashboard page

**Files:**
- Modify: `frontend/src/pages/Dashboard.tsx`

- [ ] **Step 1: Implement Dashboard**

Replace `frontend/src/pages/Dashboard.tsx`:
```tsx
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import PageShell from "../components/layout/PageShell";
import SmilesInput from "../components/SmilesInput";
import { getHistory, clearHistory, type HistoryEntry } from "../lib/history";
import { formatNum } from "../lib/utils";

export default function Dashboard() {
  const navigate = useNavigate();
  const [smiles, setSmiles] = useState("");
  const [history, setHistory] = useState<HistoryEntry[]>(getHistory());

  const handleGo = () => {
    if (smiles.trim()) {
      navigate(`/predict?smiles=${encodeURIComponent(smiles)}`);
    }
  };

  return (
    <PageShell title="Dashboard">
      <div className="mb-8">
        <h2 className="text-sm font-medium text-[var(--text-muted)] uppercase tracking-wide mb-2">Quick Predict</h2>
        <div className="flex gap-3 items-end">
          <div className="flex-1">
            <SmilesInput value={smiles} onChange={setSmiles} showPreview={false} />
          </div>
          <button onClick={handleGo} disabled={!smiles.trim()}
            className="px-6 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-50">
            Go →
          </button>
        </div>
      </div>

      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-medium text-[var(--text-muted)] uppercase tracking-wide">Recent Predictions</h2>
          {history.length > 0 && (
            <button onClick={() => { clearHistory(); setHistory([]); }}
              className="text-xs text-[var(--text-muted)] hover:text-red-400">Clear</button>
          )}
        </div>

        {history.length === 0 ? (
          <p className="text-sm text-[var(--text-muted)]">No predictions yet.</p>
        ) : (
          <div className="border rounded-lg overflow-hidden" style={{ background: "var(--surface)" }}>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-[var(--text-muted)] text-xs uppercase tracking-wide">
                  <th className="text-left px-4 py-2">SMILES</th>
                  <th className="text-right px-4 py-2">Cmax</th>
                  <th className="text-right px-4 py-2">AUC</th>
                  <th className="text-right px-4 py-2">t½</th>
                  <th className="px-4 py-2" />
                </tr>
              </thead>
              <tbody>
                {history.map((h, i) => (
                  <tr key={i} className="border-b last:border-0 hover:bg-white/5 cursor-pointer"
                    onClick={() => navigate(`/predict?smiles=${encodeURIComponent(h.smiles)}&dose=${h.dose}&route=${h.route}`)}>
                    <td className="px-4 py-2 font-mono text-xs max-w-[300px] truncate">{h.smiles}</td>
                    <td className="px-4 py-2 text-right tabular-nums">{formatNum(h.cmax)}</td>
                    <td className="px-4 py-2 text-right tabular-nums">{formatNum(h.auc)}</td>
                    <td className="px-4 py-2 text-right tabular-nums">{formatNum(h.thalf)}h</td>
                    <td className="px-4 py-2 text-right text-[var(--text-muted)]">→</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </PageShell>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/pages/Dashboard.tsx
git commit -m "feat(frontend): Dashboard with quick predict + history table"
```

---

### Task 8: Compare page

**Files:**
- Modify: `frontend/src/pages/Compare.tsx`

- [ ] **Step 1: Implement Compare page**

Replace `frontend/src/pages/Compare.tsx`:
```tsx
import { useState } from "react";
import { useQueries } from "@tanstack/react-query";
import PageShell from "../components/layout/PageShell";
import SmilesInput from "../components/SmilesInput";
import PKCurve from "../components/charts/PKCurve";
import ErrorAlert from "../components/ErrorAlert";
import { api } from "../api/client";
import { formatNum } from "../lib/utils";

const COLORS = ["#3b82f6", "#f97316", "#22c55e"];
const MAX_DRUGS = 3;

export default function Compare() {
  const [inputs, setInputs] = useState([{ smiles: "", dose: 100 }]);
  const [submitted, setSubmitted] = useState(false);

  const queries = useQueries({
    queries: inputs.map((inp, idx) => ({
      queryKey: ["predict", inp.smiles, inp.dose, "oral"],
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        api.predict({ smiles: inp.smiles, dose_mg: inp.dose, route: "oral" }, signal),
      enabled: submitted && inp.smiles.trim().length > 0,
    })),
  });

  const addDrug = () => {
    if (inputs.length < MAX_DRUGS) setInputs([...inputs, { smiles: "", dose: 100 }]);
  };
  const removeDrug = (idx: number) => {
    setInputs(inputs.filter((_, i) => i !== idx));
    setSubmitted(false);
  };
  const updateInput = (idx: number, field: "smiles" | "dose", value: string | number) => {
    const copy = [...inputs];
    (copy[idx] as Record<string, string | number>)[field] = value;
    setInputs(copy);
  };

  const isLoading = queries.some((q) => q.isLoading);
  const results = queries.map((q) => q.data).filter(Boolean);
  const errors = queries.filter((q) => q.error).map((q) => (q.error as Error).message);

  return (
    <PageShell title="Compare">
      <div className="space-y-3 mb-4">
        {inputs.map((inp, idx) => (
          <div key={idx} className="flex gap-3 items-end">
            <div className="flex-1">
              <SmilesInput value={inp.smiles} onChange={(v) => updateInput(idx, "smiles", v)} showPreview={false} />
            </div>
            <div>
              <label className="block text-xs text-[var(--text-muted)] mb-1">Dose</label>
              <input type="number" value={inp.dose} onChange={(e) => updateInput(idx, "dose", Number(e.target.value))}
                className="w-20 px-2 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]" />
            </div>
            {inputs.length > 1 && (
              <button onClick={() => removeDrug(idx)} className="text-red-400 text-sm px-2 py-2">✕</button>
            )}
          </div>
        ))}
      </div>

      <div className="flex gap-3 mb-6">
        {inputs.length < MAX_DRUGS && (
          <button onClick={addDrug} className="text-sm text-blue-400 hover:text-blue-300">+ Add Drug</button>
        )}
        <button onClick={() => setSubmitted(true)} disabled={isLoading || inputs.every((i) => !i.smiles.trim())}
          className="px-6 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-50">
          {isLoading ? "Comparing..." : "Compare"}
        </button>
      </div>

      {errors.length > 0 && errors.map((e, i) => <ErrorAlert key={i} message={e} />)}

      {results.length > 0 && (
        <div className="space-y-6">
          <PKCurve
            datasets={results.map((r, idx) => ({
              time: r!.time_h,
              conc: r!.cp_mg_L,
              label: r!.drug_name || inputs[idx]?.smiles.slice(0, 15) || `Drug ${idx + 1}`,
              color: COLORS[idx],
            }))}
          />

          <div className="border rounded-lg overflow-hidden" style={{ background: "var(--surface)" }}>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-[var(--text-muted)] text-xs uppercase tracking-wide">
                  <th className="text-left px-4 py-2">Metric</th>
                  {results.map((r, idx) => (
                    <th key={idx} className="text-right px-4 py-2" style={{ color: COLORS[idx] }}>
                      {r!.drug_name || `Drug ${idx + 1}`}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  { label: "Cmax (mg/L)", key: "cmax_mg_L" },
                  { label: "AUC (mg·h/L)", key: "auc0t_mg_h_L" },
                  { label: "t½ (h)", key: "t_half_h" },
                  { label: "Tmax (h)", key: "tmax_h" },
                ].map((row) => (
                  <tr key={row.key} className="border-b last:border-0">
                    <td className="px-4 py-2 text-[var(--text-muted)]">{row.label}</td>
                    {results.map((r, idx) => (
                      <td key={idx} className="px-4 py-2 text-right tabular-nums font-medium">
                        {formatNum((r as Record<string, number>)[row.key])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </PageShell>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/pages/Compare.tsx
git commit -m "feat(frontend): Compare page with C(t) overlay + metrics table"
```

---

### Task 9: dev.sh startup script

**Files:**
- Create: `frontend/dev.sh`

- [ ] **Step 1: Create dev script**

Create `frontend/dev.sh`:
```bash
#!/bin/bash
# Start FastAPI backend + Vite frontend dev server
set -e
cd "$(dirname "$0")/.."

echo "Starting Omega backend on :8000..."
source .venv/bin/activate
omega serve start --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Starting Vite dev server on :5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
echo "Backend PID=$BACKEND_PID, Frontend PID=$FRONTEND_PID"
echo "Open http://localhost:5173"
wait
```

```bash
chmod +x frontend/dev.sh
```

- [ ] **Step 2: Commit**

```bash
git add frontend/dev.sh
git commit -m "feat(frontend): dev.sh startup script for backend + frontend"
```

---

## Chunk 4: Remaining Pages (DDI, Dose Optimizer, PopPK, Reports)

### Task 10: Dose Optimizer page

**Files:**
- Modify: `frontend/src/pages/DoseOptimizer.tsx`

- [ ] **Step 1: Implement Dose Optimizer**

Replace `frontend/src/pages/DoseOptimizer.tsx` with the two-step flow:
1. Predict ADME via `/predict/full`
2. Convert to DrugRequest → call `/dose/optimize`
3. Re-predict with optimal dose for C(t) curve

Core logic:
```tsx
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import PageShell from "../components/layout/PageShell";
import SmilesInput from "../components/SmilesInput";
import PKCurve from "../components/charts/PKCurve";
import ErrorAlert from "../components/ErrorAlert";
import { api } from "../api/client";
import { admeToDrugRequest } from "../api/orchestrate";
import { formatNum } from "../lib/utils";

export default function DoseOptimizer() {
  const [smiles, setSmiles] = useState("");
  const [mec, setMec] = useState(1);
  const [mtc, setMtc] = useState(10);
  const [doseMin, setDoseMin] = useState(10);
  const [doseMax, setDoseMax] = useState(500);

  const mutation = useMutation({
    mutationFn: async () => {
      // Step 1: predict ADME
      const pred = await api.predict({ smiles, dose_mg: doseMin, route: "oral" });
      const drugReq = admeToDrugRequest(pred, doseMin);
      // Step 2: optimize
      const opt = await api.doseOptimize({
        drug: drugReq,
        cmin_mg_L: mec,
        cmax_mg_L: mtc,
        dose_range_mg: [doseMin, doseMax],
      });
      // Step 3: re-predict with optimal dose for curve
      const curve = await api.predict({ smiles, dose_mg: opt.optimal_dose_mg, route: "oral" });
      return { opt, curve };
    },
  });

  return (
    <PageShell title="Dose Optimizer">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="col-span-2">
          <SmilesInput value={smiles} onChange={setSmiles} showPreview={false} />
        </div>
        <div>
          <label className="block text-xs text-[var(--text-muted)] mb-1">MEC (mg/L)</label>
          <input type="number" value={mec} onChange={(e) => setMec(Number(e.target.value))}
            className="w-full px-3 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]" />
        </div>
        <div>
          <label className="block text-xs text-[var(--text-muted)] mb-1">MTC (mg/L)</label>
          <input type="number" value={mtc} onChange={(e) => setMtc(Number(e.target.value))}
            className="w-full px-3 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]" />
        </div>
        <div>
          <label className="block text-xs text-[var(--text-muted)] mb-1">Min dose (mg)</label>
          <input type="number" value={doseMin} onChange={(e) => setDoseMin(Number(e.target.value))}
            className="w-full px-3 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]" />
        </div>
        <div>
          <label className="block text-xs text-[var(--text-muted)] mb-1">Max dose (mg)</label>
          <input type="number" value={doseMax} onChange={(e) => setDoseMax(Number(e.target.value))}
            className="w-full px-3 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]" />
        </div>
      </div>

      <button onClick={() => mutation.mutate()} disabled={mutation.isPending || !smiles.trim()}
        className="px-6 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-50 mb-6">
        {mutation.isPending ? "Optimizing..." : "Optimize"}
      </button>

      {mutation.error && <ErrorAlert message={(mutation.error as Error).message} />}

      {mutation.data && (
        <div className="space-y-6">
          <div className="border rounded-lg p-4 text-center" style={{ background: "var(--surface)" }}>
            <div className="text-xs text-[var(--text-muted)] uppercase tracking-wide mb-1">Recommended Dose</div>
            <div className="text-3xl font-bold text-blue-400 tabular-nums">{mutation.data.opt.optimal_dose_mg} mg</div>
            <div className="text-xs text-[var(--text-muted)] mt-2">
              Css_max: {formatNum(mutation.data.opt.css_max)} | Css_min: {formatNum(mutation.data.opt.css_min)} | AUC_ss: {formatNum(mutation.data.opt.auc_ss)}
            </div>
          </div>

          <PKCurve
            datasets={[{
              time: mutation.data.curve.time_h,
              conc: mutation.data.curve.cp_mg_L,
              label: `${mutation.data.opt.optimal_dose_mg}mg`,
              color: "#3b82f6",
            }]}
            mecLine={mec}
            mtcLine={mtc}
          />
        </div>
      )}
    </PageShell>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/pages/DoseOptimizer.tsx
git commit -m "feat(frontend): Dose Optimizer page with MEC/MTC therapeutic window"
```

---

### Task 11: DDI Checker page

**Files:**
- Modify: `frontend/src/pages/DDIChecker.tsx`

- [ ] **Step 1: Implement DDI Checker**

Replace `frontend/src/pages/DDIChecker.tsx`. Victim uses SMILES → predict → DrugRequest orchestration. Perpetrator is manual parameter entry:

```tsx
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import PageShell from "../components/layout/PageShell";
import SmilesInput from "../components/SmilesInput";
import PKCurve from "../components/charts/PKCurve";
import ErrorAlert from "../components/ErrorAlert";
import { api } from "../api/client";
import { admeToDrugRequest } from "../api/orchestrate";

export default function DDIChecker() {
  const [victimSmiles, setVictimSmiles] = useState("");
  const [victimDose, setVictimDose] = useState(100);
  const [perpName, setPerpName] = useState("");
  const [perpEnzyme, setPerpEnzyme] = useState("CYP3A4");
  const [perpKi, setPerpKi] = useState(1);
  const [perpMechanism, setPerpMechanism] = useState("competitive");
  const [perpCmax, setPerpCmax] = useState<number | "">("");

  const mutation = useMutation({
    mutationFn: async () => {
      const pred = await api.predict({ smiles: victimSmiles, dose_mg: victimDose, route: "oral" });
      const drugReq = admeToDrugRequest(pred, victimDose);
      return api.ddiSimulate({
        victim_drug: drugReq,
        perpetrator_name: perpName,
        perpetrator_ki_uM: perpKi,
        perpetrator_target_enzyme: perpEnzyme,
        perpetrator_mechanism: perpMechanism,
        perpetrator_cmax_uM: perpCmax || undefined,
      });
    },
  });

  return (
    <PageShell title="DDI Checker">
      <div className="space-y-4 mb-6">
        <div>
          <h3 className="text-sm font-medium mb-2">Victim Drug</h3>
          <div className="flex gap-3 items-end">
            <div className="flex-1">
              <SmilesInput value={victimSmiles} onChange={setVictimSmiles} showPreview={false} />
            </div>
            <div>
              <label className="block text-xs text-[var(--text-muted)] mb-1">Dose (mg)</label>
              <input type="number" value={victimDose} onChange={(e) => setVictimDose(Number(e.target.value))}
                className="w-24 px-3 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]" />
            </div>
          </div>
        </div>

        <div>
          <h3 className="text-sm font-medium mb-2">Perpetrator</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div>
              <label className="block text-xs text-[var(--text-muted)] mb-1">Name</label>
              <input type="text" value={perpName} onChange={(e) => setPerpName(e.target.value)} placeholder="e.g. ketoconazole"
                className="w-full px-3 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]" />
            </div>
            <div>
              <label className="block text-xs text-[var(--text-muted)] mb-1">CYP Target</label>
              <select value={perpEnzyme} onChange={(e) => setPerpEnzyme(e.target.value)}
                className="w-full px-3 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]">
                {["CYP3A4", "CYP2D6", "CYP2C9", "CYP2C19", "CYP1A2"].map((e) => <option key={e}>{e}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-xs text-[var(--text-muted)] mb-1">Ki (µM)</label>
              <input type="number" value={perpKi} onChange={(e) => setPerpKi(Number(e.target.value))} step={0.1}
                className="w-full px-3 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]" />
            </div>
            <div>
              <label className="block text-xs text-[var(--text-muted)] mb-1">Mechanism</label>
              <select value={perpMechanism} onChange={(e) => setPerpMechanism(e.target.value)}
                className="w-full px-3 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]">
                <option value="competitive">Competitive</option>
                <option value="noncompetitive">Non-competitive</option>
                <option value="mechanism_based">Mechanism-based</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      <button onClick={() => mutation.mutate()} disabled={mutation.isPending || !victimSmiles.trim()}
        className="px-6 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-50 mb-6">
        {mutation.isPending ? "Checking..." : "Check DDI"}
      </button>

      {mutation.error && <ErrorAlert message={(mutation.error as Error).message} />}

      {mutation.data && (
        <div className="space-y-6">
          <div className="border rounded-lg p-4" style={{ background: "var(--surface)" }}>
            <span className="text-sm text-[var(--text-muted)]">AUC Ratio: </span>
            <span className="text-2xl font-bold tabular-nums">{mutation.data.auc_ratio.toFixed(2)}x</span>
            <span className={`ml-3 text-sm font-medium ${
              mutation.data.auc_ratio > 5 ? "text-red-400" :
              mutation.data.auc_ratio > 2 ? "text-yellow-400" : "text-green-400"
            }`}>
              {mutation.data.interaction_magnitude}
            </span>
          </div>

          {mutation.data.time_h && (
            <PKCurve
              datasets={[
                { time: mutation.data.time_h, conc: mutation.data.cp_alone, label: "Alone", color: "#3b82f6" },
                { time: mutation.data.time_h, conc: mutation.data.cp_with_inhibitor, label: "With inhibitor", color: "#ef4444" },
              ]}
            />
          )}
        </div>
      )}
    </PageShell>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/pages/DDIChecker.tsx
git commit -m "feat(frontend): DDI Checker page with victim/perpetrator interaction"
```

---

### Task 12: Population PK page

**Files:**
- Modify: `frontend/src/pages/PopulationPK.tsx`

- [ ] **Step 1: Implement PopulationPK (summary stats view)**

Replace `frontend/src/pages/PopulationPK.tsx`:
```tsx
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import PageShell from "../components/layout/PageShell";
import SmilesInput from "../components/SmilesInput";
import ErrorAlert from "../components/ErrorAlert";
import { api } from "../api/client";
import { admeToDrugRequest } from "../api/orchestrate";
import { formatNum } from "../lib/utils";

export default function PopulationPK() {
  const [smiles, setSmiles] = useState("");
  const [dose, setDose] = useState(100);
  const [nSubjects, setNSubjects] = useState(100);

  const mutation = useMutation({
    mutationFn: async () => {
      const pred = await api.predict({ smiles, dose_mg: dose, route: "oral" });
      const drugReq = admeToDrugRequest(pred, dose);
      return api.population({ drug: drugReq, n_subjects: nSubjects, dose_mg: dose, route: "oral" });
    },
  });

  return (
    <PageShell title="Population PK">
      <div className="flex gap-4 items-end mb-6">
        <div className="flex-1">
          <SmilesInput value={smiles} onChange={setSmiles} showPreview={false} />
        </div>
        <div>
          <label className="block text-xs text-[var(--text-muted)] mb-1">Dose (mg)</label>
          <input type="number" value={dose} onChange={(e) => setDose(Number(e.target.value))}
            className="w-24 px-3 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]" />
        </div>
        <div>
          <label className="block text-xs text-[var(--text-muted)] mb-1">N subjects</label>
          <input type="number" value={nSubjects} onChange={(e) => setNSubjects(Number(e.target.value))}
            className="w-24 px-3 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]" />
        </div>
        <button onClick={() => mutation.mutate()} disabled={mutation.isPending || !smiles.trim()}
          className="px-6 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-50">
          {mutation.isPending ? "Simulating..." : "Simulate"}
        </button>
      </div>

      {mutation.error && <ErrorAlert message={(mutation.error as Error).message} />}

      {mutation.data && (
        <div className="border rounded-lg overflow-hidden" style={{ background: "var(--surface)" }}>
          <div className="px-4 py-3 border-b text-xs font-medium uppercase tracking-wide text-[var(--text-muted)]">
            Population Summary (N={mutation.data.n_subjects})
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-[var(--text-muted)] text-xs uppercase">
                <th className="text-left px-4 py-2">Metric</th>
                <th className="text-right px-4 py-2">Median</th>
                <th className="text-right px-4 py-2">Mean</th>
                <th className="text-right px-4 py-2">CV%</th>
                <th className="text-right px-4 py-2">P5</th>
                <th className="text-right px-4 py-2">P95</th>
              </tr>
            </thead>
            <tbody>
              {[
                { label: "Cmax (mg/L)", median: mutation.data.cmax_median, mean: mutation.data.cmax_mean, cv: mutation.data.cmax_cv_pct, p5: mutation.data.cmax_p5, p95: mutation.data.cmax_p95 },
                { label: "AUC (mg·h/L)", median: mutation.data.auc_median, mean: mutation.data.auc_mean, cv: mutation.data.auc_cv_pct, p5: 0, p95: 0 },
              ].map((row) => (
                <tr key={row.label} className="border-b last:border-0">
                  <td className="px-4 py-2 text-[var(--text-muted)]">{row.label}</td>
                  <td className="px-4 py-2 text-right tabular-nums font-medium">{formatNum(row.median)}</td>
                  <td className="px-4 py-2 text-right tabular-nums">{formatNum(row.mean)}</td>
                  <td className="px-4 py-2 text-right tabular-nums">{formatNum(row.cv)}%</td>
                  <td className="px-4 py-2 text-right tabular-nums">{formatNum(row.p5)}</td>
                  <td className="px-4 py-2 text-right tabular-nums">{formatNum(row.p95)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </PageShell>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/pages/PopulationPK.tsx
git commit -m "feat(frontend): Population PK page with summary statistics"
```

---

### Task 13: Reports page

**Files:**
- Modify: `frontend/src/pages/Reports.tsx`

- [ ] **Step 1: Implement Reports page**

Replace `frontend/src/pages/Reports.tsx`:
```tsx
import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import PageShell from "../components/layout/PageShell";
import SmilesInput from "../components/SmilesInput";
import ErrorAlert from "../components/ErrorAlert";
import { api } from "../api/client";
import { downloadBlob } from "../lib/utils";

export default function Reports() {
  const [smiles, setSmiles] = useState("");
  const [drugName, setDrugName] = useState("");
  const [dose, setDose] = useState(100);
  const [route, setRoute] = useState("oral");
  const [previewHtml, setPreviewHtml] = useState("");

  const mutation = useMutation({
    mutationFn: async () => {
      const res = await api.report({ smiles, drug_name: drugName || "compound", dose_mg: dose, route });
      const html = atob(res.data);
      setPreviewHtml(html);
      return html;
    },
  });

  return (
    <PageShell title="Reports">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="col-span-2">
          <SmilesInput value={smiles} onChange={setSmiles} showPreview={false} />
        </div>
        <div>
          <label className="block text-xs text-[var(--text-muted)] mb-1">Drug Name</label>
          <input type="text" value={drugName} onChange={(e) => setDrugName(e.target.value)} placeholder="Optional"
            className="w-full px-3 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]" />
        </div>
        <div className="flex gap-2">
          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1">Dose</label>
            <input type="number" value={dose} onChange={(e) => setDose(Number(e.target.value))}
              className="w-20 px-3 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]" />
          </div>
          <div>
            <label className="block text-xs text-[var(--text-muted)] mb-1">Route</label>
            <select value={route} onChange={(e) => setRoute(e.target.value)}
              className="px-3 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]">
              <option value="oral">Oral</option>
              <option value="iv">IV</option>
            </select>
          </div>
        </div>
      </div>

      <div className="flex gap-3 mb-6">
        <button onClick={() => mutation.mutate()} disabled={mutation.isPending || !smiles.trim()}
          className="px-6 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-50">
          {mutation.isPending ? "Generating..." : "Generate Report"}
        </button>
        {previewHtml && (
          <button onClick={() => downloadBlob(previewHtml, `${drugName || "report"}.html`, "text/html")}
            className="px-4 py-2 rounded-md border text-sm text-[var(--text-muted)] hover:text-[var(--text)]">
            📥 Download HTML
          </button>
        )}
      </div>

      {mutation.error && <ErrorAlert message={(mutation.error as Error).message} />}

      {previewHtml && (
        <div className="border rounded-lg overflow-hidden" style={{ background: "#fff" }}>
          <iframe
            srcDoc={previewHtml}
            className="w-full"
            style={{ height: "600px", border: "none" }}
            title="Report Preview"
          />
        </div>
      )}
    </PageShell>
  );
}
```

- [ ] **Step 2: Commit**

```bash
git add frontend/src/pages/Reports.tsx
git commit -m "feat(frontend): Reports page with HTML preview + download"
```

---

### Task 14: Final integration test + cleanup

- [ ] **Step 1: Remove Vite boilerplate files**

```bash
rm -f frontend/src/App.css frontend/src/assets/react.svg frontend/public/vite.svg
```

- [ ] **Step 2: Update `index.html` title and dark class**

Edit `frontend/index.html` — add `class="dark"` to `<html>` tag, update `<title>` to "Omega PBPK".

- [ ] **Step 3: Full smoke test**

Run `frontend/dev.sh`. Visit each page:
1. `/` — Dashboard shows, quick predict navigates to /predict
2. `/predict` — Enter caffeine SMILES, predict, see results
3. `/compare` — Add 2 drugs, compare, see overlay chart
4. `/dose-optimize` — Enter SMILES + MEC/MTC, optimize
5. `/ddi` — Enter victim + perpetrator params, check DDI
6. `/population` — Enter SMILES, simulate population
7. `/reports` — Generate and preview report

Sidebar navigation works. Status bar shows API connected.

- [ ] **Step 4: Final commit**

```bash
git add -A frontend/
git commit -m "feat(frontend): complete Omega PBPK React frontend (7 pages)"
```
