import { useState, useEffect, useCallback } from "react";
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
import type { CardData } from "../components/charts/PKCards";
import type { Dataset } from "../components/charts/PKCurve";
import type { PredictRequest } from "../api/types";

function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded bg-[var(--border)] ${className}`} />;
}

export default function Predict() {
  const [searchParams] = useSearchParams();

  const [smiles, setSmiles] = useState(searchParams.get("smiles") ?? "");
  const [dose, setDose] = useState(searchParams.get("dose") ?? "100");
  const [route, setRoute] = useState(searchParams.get("route") ?? "oral");
  const [submitted, setSubmitted] = useState(false);

  const request: PredictRequest = {
    smiles: smiles.trim(),
    dose_mg: Number(dose) || 100,
    route,
  };

  const { data, error, isFetching, refetch } = useQuery({
    queryKey: ["predict", request.smiles, request.dose_mg, request.route],
    queryFn: ({ signal }) => api.predict(request, signal),
    enabled: false,
    retry: false,
  });

  const handlePredict = useCallback(() => {
    if (!smiles.trim()) return;
    setSubmitted(true);
    refetch();
  }, [smiles, refetch]);

  // Ctrl+Enter shortcut
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        handlePredict();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handlePredict]);

  // Auto-save to history on successful prediction
  useEffect(() => {
    if (data) {
      addToHistory({
        smiles: data.smiles,
        dose: Number(dose) || 100,
        route,
        cmax: data.cmax_mg_L,
        auc: data.auc0t_mg_h_L,
        thalf: data.t_half_h,
        timestamp: Date.now(),
      });
    }
  }, [data, dose, route]);

  const cards: CardData[] | null = data
    ? [
        { label: "Cmax", value: data.cmax_mg_L, unit: "mg/L", p5: data.cmax_p5, p95: data.cmax_p95 },
        { label: "Tmax", value: data.tmax_h, unit: "h" },
        { label: "AUC\u2080\u208B\u209C", value: data.auc0t_mg_h_L, unit: "mg\u00B7h/L", p5: data.auc_p5, p95: data.auc_p95 },
        { label: "t\u00BD", value: data.t_half_h, unit: "h" },
      ]
    : null;

  const datasets: Dataset[] | null = data
    ? [{ time: data.time_h, conc: data.cp_mg_L, label: data.drug_name || "Compound", color: "#3b82f6" }]
    : null;

  return (
    <PageShell title="Predict">
      {/* Input section */}
      <div className="space-y-4 mb-8">
        <SmilesInput value={smiles} onChange={setSmiles} />

        <div className="flex items-end gap-4">
          <div>
            <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">
              Dose (mg)
            </label>
            <input
              type="number"
              value={dose}
              onChange={(e) => setDose(e.target.value)}
              min={0}
              className="w-28 rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] focus:outline-none focus:ring-1 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">
              Route
            </label>
            <select
              value={route}
              onChange={(e) => setRoute(e.target.value)}
              className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] focus:outline-none focus:ring-1 focus:ring-blue-500"
            >
              <option value="oral">Oral</option>
              <option value="iv">IV Bolus</option>
              <option value="iv_infusion">IV Infusion</option>
            </select>
          </div>

          <button
            onClick={handlePredict}
            disabled={!smiles.trim() || isFetching}
            className="px-5 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {isFetching ? "Predicting..." : "Predict"}
          </button>

          <span className="text-xs text-[var(--text-muted)] self-center">Ctrl+Enter</span>
        </div>
      </div>

      {/* Error */}
      {error && <ErrorAlert message={(error as Error).message} />}

      {/* Loading skeleton */}
      {isFetching && (
        <div className="space-y-6">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-24" />
            ))}
          </div>
          <Skeleton className="h-80" />
          <Skeleton className="h-48" />
        </div>
      )}

      {/* Results */}
      {!isFetching && data && submitted && (
        <div className="space-y-6">
          {/* Warnings */}
          {data.warnings.length > 0 && (
            <WarningBadge warnings={data.warnings} riskLevel={data.overall_risk_level} />
          )}

          {/* PK Cards */}
          {cards && <PKCards cards={cards} confidence={data.confidence} />}

          {/* PK Curve */}
          {datasets && <PKCurve datasets={datasets} />}

          {/* ADME Table */}
          {data.adme && <ADMETable adme={data.adme} />}
        </div>
      )}
    </PageShell>
  );
}
