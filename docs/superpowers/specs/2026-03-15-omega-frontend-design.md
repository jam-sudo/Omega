# Omega PBPK Frontend — Design Spec

> Date: 2026-03-15
> Status: Approved (post-review)
> Author: jam + Claude

## Overview

Modern web frontend for the Omega PBPK platform. Replaces the existing minimal HTML/JS/CSS frontend with a full React SPA backed by the existing FastAPI REST API (40+ endpoints).

**Goal**: Personal research dashboard for pharmacokinetic prediction, drug comparison, dose optimization, and population simulation.

**User**: Single user (developer/researcher). No auth, no multi-tenancy.

## Tech Stack

| Layer | Choice | Rationale |
|-------|--------|-----------|
| Framework | React 18 + Vite | Fast HMR, lightweight SPA, no SSR needed |
| Language | TypeScript | Type safety, autocomplete |
| Styling | Tailwind CSS | Rapid prototyping, utility-first |
| Components | shadcn/ui (Radix-based) | Clean, dark-mode, copy-paste ownership |
| Charts | Recharts | Declarative, good for line/area charts with confidence bands |
| API State | TanStack Query | Caching, deduplication, loading/error states |
| Molecule | smiles-drawer | Lightweight 2D structure preview (best-effort; may fail on complex ring systems) |
| API Types | openapi-typescript (dev) | Auto-generate types from FastAPI OpenAPI schema |
| Backend | Existing FastAPI (`omega serve`) | 40+ endpoints, no changes needed |

## Architecture

### Project Structure

```
frontend/
├── index.html
├── package.json
├── vite.config.ts              # Proxy: /api → localhost:8000 (strip prefix)
├── tailwind.config.ts
├── tsconfig.json
├── dev.sh                       # Starts FastAPI + Vite together
├── src/
│   ├── main.tsx                 # React root + QueryClientProvider
│   ├── App.tsx                  # Router + sidebar layout
│   ├── api/
│   │   ├── client.ts            # Typed fetch wrapper per endpoint
│   │   ├── orchestrate.ts       # SMILES → predict → DrugRequest conversion
│   │   └── types.ts             # Auto-generated from /openapi.json
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx      # Collapsible sidebar navigation
│   │   │   ├── PageShell.tsx    # Page wrapper (title, breadcrumb, <title>)
│   │   │   └── StatusBar.tsx    # Bottom bar: API status, latency, version
│   │   ├── charts/
│   │   │   ├── PKCurve.tsx      # C(t) line chart (single/multi/band modes)
│   │   │   ├── PKCards.tsx      # 4 metric cards (Cmax, Tmax, AUC, t½)
│   │   │   └── PopulationBand.tsx # VPC-style percentile bands
│   │   ├── SmilesInput.tsx      # SMILES text input + structure preview
│   │   ├── ADMETable.tsx        # ADME properties + confidence intervals
│   │   ├── ErrorAlert.tsx       # Inline error display (red alert box)
│   │   ├── WarningBadge.tsx     # Risk flag / warning display
│   │   └── ui/                  # shadcn/ui primitives
│   ├── pages/
│   │   ├── Dashboard.tsx        # Home: quick predict + history table
│   │   ├── Predict.tsx          # Core: SMILES → full PK prediction
│   │   ├── Compare.tsx          # Side-by-side drug comparison (2-3)
│   │   ├── DoseOptimizer.tsx    # Therapeutic window optimization
│   │   ├── DDIChecker.tsx       # Drug-drug interaction simulation
│   │   ├── PopulationPK.tsx     # Virtual patient population (summary stats)
│   │   └── Reports.tsx          # Generate + download HTML reports
│   └── lib/
│       ├── utils.ts             # Formatting, unit conversion
│       └── history.ts           # localStorage prediction history
```

### Routing

| Path | Page | API Flow |
|------|------|----------|
| `/` | Dashboard | localStorage |
| `/predict` | Predict | `POST /predict/full` |
| `/compare` | Compare | `POST /predict/full` × N (or `POST /compare`) |
| `/dose-optimize` | DoseOptimizer | `POST /predict/full` → build DrugRequest → `POST /dose/optimize` |
| `/ddi` | DDIChecker | `POST /predict/full` (victim) → build DrugRequest → `POST /ddi/simulate` |
| `/population` | PopulationPK | `POST /predict/full` → build DrugRequest → `POST /population` |
| `/reports` | Reports | `POST /report` (returns base64 JSON, decode client-side) |

### API Orchestration Pattern

DDI, Dose Optimizer, and Population PK endpoints require a `DrugRequest` object (not SMILES). The frontend orchestrates a two-step flow:

```
Step 1: POST /predict/full { smiles } → FullPredictResponse (ADME properties)
Step 2: Convert FullPredictResponse.adme → DrugRequest { name, mw, logP, fup, ... }
Step 3: POST /ddi/simulate { victim_drug: DrugRequest, ... }
```

This conversion lives in `api/orchestrate.ts`:
```typescript
function admeTooDrugRequest(adme: AdmeProperties, name: string): DrugRequest
```

TanStack Query caches the `/predict/full` result, so Step 1 is instant if the drug was previously predicted on any page.

### Backend Connection

- Vite dev proxy: `/api/*` → `http://localhost:8000/*` (strips `/api` prefix)
- Production: `npm run build` → `dist/` → served from FastAPI static files
- `dev.sh` starts both servers: `omega serve start & cd frontend && npm run dev`
- CORS: handled by Vite proxy in dev; same-origin in production

## Page Designs

### Dashboard

Quick-access home page.

```
┌──────────────────────────────────────────┐
│  Quick Predict                           │
│  [SMILES ──────────────────] [▶ Go]     │
├──────────────────────────────────────────┤
│  Recent Predictions                      │
│  ┌───────┬───────┬──────┬──────┬────┐   │
│  │ SMILES│ Cmax  │ AUC  │ t½   │ →  │   │
│  │ Cn1c… │ 1.36  │ 9.39 │ 8.2h │ →  │   │
│  │ COCCc…│ 0.15  │ 0.52 │ 6.7h │ →  │   │
│  └───────┴───────┴──────┴──────┴────┘   │
│  → links to /predict with pre-filled     │
└──────────────────────────────────────────┘
```

- History stored in localStorage (last 20 predictions, summary only)
- Click row → navigate to `/predict?smiles=...&dose=...`

### Predict (Core Page)

```
┌──────────────────────────────────────────────────┐
│  Predict                                          │
├──────────────────────────────────────────────────┤
│  [SMILES ─────────────────] [Dose: 100mg]        │
│  [2D structure preview]     [Route: ▾oral]       │
│                              [🔵 Predict]         │
├──────────────────────────────────────────────────┤
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────┐│
│  │ Cmax     │ │ Tmax     │ │ AUC      │ │ t½   ││
│  │ 15.7     │ │ 2.5h     │ │ 98.3     │ │ 4.2h ││
│  │ mg/L     │ │          │ │ mg·h/L   │ │      ││
│  │ ▓▓▓▓▓░░░ │ │          │ │          │ │      ││
│  │ CI:8-23  │ │          │ │          │ │      ││
│  └──────────┘ └──────────┘ └──────────┘ └──────┘│
├──────────────────────────────────────────────────┤
│  C(t) Curve                    [linear | log]    │
│  ┌──────────────────────────────────────┐        │
│  │  Recharts LineChart                  │        │
│  │  - Hover tooltip with concentration  │        │
│  │  - CSS draw animation on load        │        │
│  └──────────────────────────────────────┘        │
│  [📋 Copy] [📥 CSV] [🖼 PNG]                     │
├──────────────────┬───────────────────────────────┤
│  ADME Properties │  Warnings / Risk              │
│  MW:  309 g/mol  │  ⚠ High lipophilicity         │
│  logP: 4.44      │  ⚠ Low solubility             │
│  fup: 0.05 [.01] │                               │
│  peff: 2.27      │  Confidence: ●●○ medium       │
│  rbp: 0.79       │                               │
└──────────────────┴───────────────────────────────┘
```

- SMILES input: real-time 2D structure preview (smiles-drawer)
- PK Cards: value + unit + confidence bar + 90% CI
  - Props accept `FullPredictResponse` fields: `cmax_p5/p50/p95`, `auc_p5/p50/p95`
- C(t) chart: linear/semilog toggle, hover tooltip, export buttons
- ADME table: all properties with confidence intervals
- Auto-save to localStorage history on successful prediction

### Compare

```
┌──────────────────────────────────────────────────┐
│  [SMILES 1 ─────────────] [+ Add Drug]           │
│  [SMILES 2 ─────────────] [✕ Remove]             │
│  [🔵 Compare]                                     │
├──────────────────────────────────────────────────┤
│  C(t) Overlay Chart                               │
│  ─── Drug A (blue)    ─── Drug B (orange)        │
├──────────────────────────────────────────────────┤
│  Comparison Table                                 │
│  ┌──────┬──────────┬──────────┐                  │
│  │      │ Drug A   │ Drug B   │                  │
│  │ Cmax │ 15.7     │ 8.3      │                  │
│  │ AUC  │ 98.3     │ 45.1     │                  │
│  │ t½   │ 4.2h     │ 6.8h     │                  │
│  │ fup  │ 0.05     │ 0.12     │                  │
│  └──────┴──────────┴──────────┘                  │
└──────────────────────────────────────────────────┘
```

- Max 3 drugs (color-coded: blue, orange, green)
- Uses `POST /predict/full` × N (sequential or `POST /compare` if available)
- PKCurve component in multi-dataset mode
- TanStack Query caching: drugs predicted on /predict available instantly

### Dose Optimizer

Two-step flow: predict ADME first, then optimize.

```
┌──────────────────────────────────────────────────┐
│  [SMILES ─────────────]                           │
│  Target: MEC [___] mg/L    MTC [___] mg/L        │
│  Dose range: [10] to [500] mg                     │
│  [🔵 Optimize]                                    │
├──────────────────────────────────────────────────┤
│  Recommended Dose: 200 mg                         │
│  Steady-state metrics: Css_max, Css_min, AUC_ss  │
│  C(t) chart: simulate recommended dose via        │
│  /predict/full with optimized dose → show curve   │
│  with MEC/MTC horizontal reference lines          │
└──────────────────────────────────────────────────┘
```

- Step 1: `/predict/full` → ADME → DrugRequest
- Step 2: `/dose/optimize` → recommended dose + steady-state metrics
- Step 3: `/predict/full` with optimized dose → C(t) curve for visualization

### DDI Checker

Victim is SMILES-based; perpetrator is parameter-based (Ki, mechanism).

```
┌──────────────────────────────────────────────────┐
│  Victim Drug:                                     │
│    [SMILES ──────────────────] [100mg]            │
│                                                    │
│  Perpetrator:                                      │
│    Name: [─────────]                               │
│    CYP Target: [▾3A4]                             │
│    Ki: [___] µM                                    │
│    Mechanism: [▾competitive]                       │
│    Cmax: [___] µM  (optional)                      │
│  [🔵 Check DDI]                                   │
├──────────────────────────────────────────────────┤
│  AUC Ratio: 2.8x   ⚠ Moderate interaction        │
│  C(t): alone (solid) vs with inhibitor (dashed)   │
└──────────────────────────────────────────────────┘
```

- Victim: SMILES → `/predict/full` → DrugRequest → `/ddi/simulate`
- Perpetrator: manual parameter entry (Ki, mechanism, Cmax), NOT SMILES

### Population PK

Summary statistics view (backend does not return per-timepoint percentiles).

```
┌──────────────────────────────────────────────────┐
│  [SMILES ─────────────] [100mg] [N subjects: 100]│
│  [🔵 Simulate Population]                        │
├──────────────────────────────────────────────────┤
│  Population Summary Statistics                    │
│  ┌────────┬────────┬────────┬────────┐           │
│  │        │ Median │ Mean   │ CV%    │           │
│  │ Cmax   │ 12.3   │ 13.1   │ 28%    │           │
│  │ AUC    │ 85.2   │ 91.4   │ 35%    │           │
│  │ t½     │ 4.1    │ 4.5    │ 22%    │           │
│  └────────┴────────┴────────┴────────┘           │
│  Cmax Distribution: [▁▂▃▅▇█▇▅▃▂▁] histogram     │
│  Percentiles: P5=6.2  P50=12.3  P95=22.1        │
└──────────────────────────────────────────────────┘
```

- No VPC bands (backend returns summary stats only, not time-series)
- PopulationBand component deferred to Future Extensions
- Histogram of Cmax/AUC distribution using Recharts BarChart

### Reports

```
┌──────────────────────────────────────────────────┐
│  [SMILES ─────────────] [Drug Name ──────]       │
│  [100mg] [oral]                                   │
│  ☑ ADME  ☑ NCA  ☑ DDI  ☑ PopPK                  │
│  [🔵 Generate Report]                            │
├──────────────────────────────────────────────────┤
│  Report Preview (iframe with decoded HTML)        │
│  [📥 Download HTML]                               │
└──────────────────────────────────────────────────┘
```

- API returns `{ data: "<base64-html>", encoding: "base64" }`
- Client decodes base64 → renders in iframe / blob URL
- Download button creates blob and triggers download

## Shared Components

| Component | Props | Used By |
|-----------|-------|---------|
| `SmilesInput` | `value, onChange, showPreview?` | All pages |
| `PKCards` | `response: FullPredictResponse` (extracts cmax/auc/tmax/thalf + percentiles) | Predict, Compare, DDI |
| `PKCurve` | `datasets: {time, conc, label, color}[], logScale?, mecLine?, mtcLine?` | Predict, Compare, DoseOpt, DDI |
| `ADMETable` | `adme: Record<string, Any>` (from API response directly) | Predict, Compare |
| `ErrorAlert` | `message: string` | All pages |
| `WarningBadge` | `warnings: string[], riskLevel?` | Predict, DDI |
| `Sidebar` | `collapsed, onToggle` | App layout |
| `PageShell` | `title, children` (sets document.title) | All pages |
| `StatusBar` | — (reads API health internally) | App layout |

## API Layer

### Client (`api/client.ts`)

```typescript
const API_BASE = "/api";  // Vite proxies to localhost:8000, stripping /api prefix

export const api = {
  predict:      (req) => post<FullPredictResponse>("/predict/full", req),
  compare:      (req) => post<CompareResponse>("/compare", req),
  simulate:     (req) => post<PKSummaryResponse>("/simulate", req),
  ddiSimulate:  (req) => post<DDIResponse>("/ddi/simulate", req),
  doseOptimize: (req) => post<DoseOptResponse>("/dose/optimize", req),
  population:   (req) => post<PopResponse>("/population", req),
  report:       (req) => post<ReportResponse>("/report", req),  // base64 JSON
  health:       ()    => get<HealthResponse>("/health"),
};
```

### Orchestration (`api/orchestrate.ts`)

```typescript
// Convert ADME prediction to DrugRequest for downstream endpoints
export function admeToDrugRequest(
  adme: FullPredictResponse["adme"],
  smiles: string,
  dose_mg: number,
): DrugRequest {
  return {
    name: smiles.slice(0, 20),
    mw: adme.mw,
    logP: adme.logP,
    fup: adme.fup,
    rbp: adme.rbp,
    clint_hepatic_L_per_h: adme.clint_hepatic_L_per_h ?? 0.2,
    peff: adme.peff,
    dose_mg,
  };
}
```

### TanStack Query Usage

```typescript
// Per-page pattern:
const { data, isLoading, error, refetch } = useQuery({
  queryKey: ['predict', smiles, dose, route],
  queryFn: ({ signal }) => api.predict({ smiles, dose_mg: dose, route }, signal),
  enabled: false,  // Manual trigger via refetch()
});
```

- Cache: same SMILES+dose+route → instant result across pages
- AbortController `signal` passed for request cancellation on re-submit
- No global state management needed

### Type Generation

```bash
npx openapi-typescript http://localhost:8000/openapi.json -o src/api/types.ts
```

Run once after API changes. Types stay in sync with FastAPI models.

## UX Details

### Theme
- **Dark mode only** (shadcn/ui `dark` class)
- Background: `#0a0a0a`, Surface: `#141414`, Border: `#262626`
- Accent: blue-500 (`#3b82f6`)
- Confidence colors: green (high), yellow (medium), red (low)

### Typography
- Font: Inter (body) — clean, modern
- Numbers: `font-variant-numeric: tabular-nums` for aligned data
- Hierarchy: 24px page title, 18px section, 14px body, 12px label

### Loading States
- Skeleton placeholders matching content layout (shadcn/ui `Skeleton`)
- No spinners

### Error States
- Inline `ErrorAlert` component (red border, error icon, message text)
- Shown below the submit button on the page where the error occurred
- API errors show the backend message; network errors show "API unreachable"

### Keyboard Shortcuts
- `Ctrl+K`: Focus SMILES input
- `Ctrl+Enter`: Run prediction / submit form

### Data Export
Below charts: `[📋 Copy] [📥 CSV] [🖼 PNG]`
- Copy: PK summary as tab-separated table to clipboard
- CSV: time + concentration data
- PNG: chart screenshot via canvas export

### Collapsible Sidebar
- Default: expanded (240px) on screens > 1280px
- Collapsed: icon-only rail (60px)
- Toggle button at bottom of sidebar
- Active page highlighted

### Status Bar (bottom)
```
API: ● connected  |  Last response: 73ms  |  Omega v0.1
```
- Green dot if API reachable, red if not
- Ping `/health` endpoint every 30s

### Prediction History
- Stored in localStorage under key `omega_history`
- Max 20 entries, FIFO eviction
- Stored fields: smiles, dose, route, cmax, auc, thalf, timestamp
- Full C(t) curves NOT stored (re-fetch on click)

## Development Workflow

### Setup
```bash
cd frontend
npm install
npm run generate-types  # Generate API types from OpenAPI
```

### Development
```bash
./dev.sh  # Starts FastAPI (port 8000) + Vite (port 5173)
```

### Vite Proxy Config
```typescript
// vite.config.ts
export default defineConfig({
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
});
```

### Production Build
```bash
npm run build  # Output to dist/
# Copy dist/ to src/omega_pbpk/static/ for FastAPI serving
```

## Future Extensions (Not In Scope)

- Framer Motion animations (currently CSS-only)
- RDKit.js for publication-quality molecule rendering
- PopulationBand VPC chart (needs backend to return per-timepoint percentiles)
- Authentication / multi-user
- Database backend (PostgreSQL) for persistent history
- i18n / localization
- Mobile responsive layout
