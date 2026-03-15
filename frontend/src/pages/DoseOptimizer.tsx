import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import PageShell from "../components/layout/PageShell";
import SmilesInput from "../components/SmilesInput";
import PKCurve from "../components/charts/PKCurve";
import ErrorAlert from "../components/ErrorAlert";
import { api } from "../api/client";
import { admeToDrugRequest } from "../api/orchestrate";
import { formatNum } from "../lib/utils";
import type { Dataset } from "../components/charts/PKCurve";
import type { DoseOptimizeResponse, FullPredictResponse } from "../api/types";

interface OptResult {
  optimize: DoseOptimizeResponse;
  curve: FullPredictResponse;
}

export default function DoseOptimizer() {
  const [smiles, setSmiles] = useState("");
  const [mec, setMec] = useState("0.5");
  const [mtc, setMtc] = useState("10");
  const [doseMin, setDoseMin] = useState("10");
  const [doseMax, setDoseMax] = useState("500");

  const mutation = useMutation<OptResult, Error>({
    mutationFn: async () => {
      // Step 1: predict ADME
      const adme = await api.predict({ smiles: smiles.trim(), dose_mg: 100 });
      const drug = admeToDrugRequest(adme, 100);

      // Step 2: optimize dose
      const optimize = await api.doseOptimize({
        drug,
        cmin_mg_L: Number(mec) || 0.5,
        cmax_mg_L: Number(mtc) || 10,
        dose_range_mg: [Number(doseMin) || 10, Number(doseMax) || 500],
      });

      // Step 3: re-predict with optimal dose for C(t) curve
      const curve = await api.predict({
        smiles: smiles.trim(),
        dose_mg: optimize.optimal_dose_mg,
      });

      return { optimize, curve };
    },
  });

  const handleOptimize = () => {
    if (!smiles.trim()) return;
    mutation.mutate();
  };

  const result = mutation.data;

  const datasets: Dataset[] | null = result
    ? [
        {
          time: result.curve.time_h,
          conc: result.curve.cp_mg_L,
          label: result.curve.drug_name || "Compound",
          color: "#3b82f6",
        },
      ]
    : null;

  const inputClass =
    "w-28 rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] focus:outline-none focus:ring-1 focus:ring-blue-500";

  return (
    <PageShell title="Dose Optimizer">
      <div className="space-y-4 mb-8">
        <SmilesInput value={smiles} onChange={setSmiles} />

        <div className="flex items-end gap-4 flex-wrap">
          <div>
            <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">
              MEC (mg/L)
            </label>
            <input
              type="number"
              value={mec}
              onChange={(e) => setMec(e.target.value)}
              min={0}
              step={0.1}
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">
              MTC (mg/L)
            </label>
            <input
              type="number"
              value={mtc}
              onChange={(e) => setMtc(e.target.value)}
              min={0}
              step={0.1}
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">
              Dose min (mg)
            </label>
            <input
              type="number"
              value={doseMin}
              onChange={(e) => setDoseMin(e.target.value)}
              min={0}
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">
              Dose max (mg)
            </label>
            <input
              type="number"
              value={doseMax}
              onChange={(e) => setDoseMax(e.target.value)}
              min={0}
              className={inputClass}
            />
          </div>

          <button
            onClick={handleOptimize}
            disabled={!smiles.trim() || mutation.isPending}
            className="px-5 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {mutation.isPending ? "Optimizing..." : "Optimize"}
          </button>
        </div>
      </div>

      {mutation.error && <ErrorAlert message={mutation.error.message} />}

      {mutation.isPending && (
        <div className="space-y-4">
          <div className="animate-pulse rounded bg-[var(--border)] h-24" />
          <div className="animate-pulse rounded bg-[var(--border)] h-80" />
        </div>
      )}

      {result && !mutation.isPending && (
        <div className="space-y-6">
          {/* Recommended dose + steady-state metrics */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[
              { label: "Optimal Dose", value: formatNum(result.optimize.optimal_dose_mg), unit: "mg" },
              { label: "Css,max", value: formatNum(result.optimize.css_max), unit: "mg/L" },
              { label: "Css,min", value: formatNum(result.optimize.css_min), unit: "mg/L" },
              { label: "AUCss", value: formatNum(result.optimize.auc_ss), unit: "mg\u00B7h/L" },
            ].map((card) => (
              <div
                key={card.label}
                className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4"
              >
                <p className="text-xs text-[var(--text-muted)] mb-1">{card.label}</p>
                <p className="text-xl font-semibold text-[var(--text)]">
                  {card.value}{" "}
                  <span className="text-sm font-normal text-[var(--text-muted)]">{card.unit}</span>
                </p>
              </div>
            ))}
          </div>

          {/* C(t) chart with MEC/MTC lines */}
          {datasets && (
            <PKCurve
              datasets={datasets}
              mecLine={Number(mec) || undefined}
              mtcLine={Number(mtc) || undefined}
            />
          )}
        </div>
      )}
    </PageShell>
  );
}
