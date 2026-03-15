import type {
  PredictRequest, FullPredictResponse,
  DDISimulateRequest, DDIResponse,
  DoseOptimizeRequest, DoseOptimizeResponse,
  PopulationRequest, PopulationResponse,
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
