import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import PageShell from "../components/layout/PageShell";
import SmilesInput from "../components/SmilesInput";
import ErrorAlert from "../components/ErrorAlert";
import { api } from "../api/client";
import { admeToDrugRequest } from "../api/orchestrate";
import { formatNum } from "../lib/utils";
import type { PopulationResponse } from "../api/types";

export default function PopulationPK() {
  const [smiles, setSmiles] = useState("");
  const [dose, setDose] = useState("100");
  const [nSubjects, setNSubjects] = useState("100");

  const mutation = useMutation<PopulationResponse, Error>({
    mutationFn: async () => {
      // Step 1: predict ADME
      const adme = await api.predict({
        smiles: smiles.trim(),
        dose_mg: Number(dose) || 100,
      });
      const drug = admeToDrugRequest(adme, Number(dose) || 100);

      // Step 2: population simulation
      return api.population({
        drug,
        n_subjects: Number(nSubjects) || 100,
        dose_mg: Number(dose) || 100,
      });
    },
  });

  const handleRun = () => {
    if (!smiles.trim()) return;
    mutation.mutate();
  };

  const result = mutation.data;

  const inputClass =
    "w-28 rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] focus:outline-none focus:ring-1 focus:ring-blue-500";

  const cellClass = "px-4 py-3 text-sm text-[var(--text)]";
  const headerClass = "px-4 py-3 text-xs font-medium text-[var(--text-muted)] uppercase tracking-wider";

  return (
    <PageShell title="Population PK">
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
              className={inputClass}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">
              N subjects
            </label>
            <input
              type="number"
              value={nSubjects}
              onChange={(e) => setNSubjects(e.target.value)}
              min={10}
              max={10000}
              className={inputClass}
            />
          </div>

          <button
            onClick={handleRun}
            disabled={!smiles.trim() || mutation.isPending}
            className="px-5 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {mutation.isPending ? "Simulating..." : "Run Population"}
          </button>
        </div>
      </div>

      {mutation.error && <ErrorAlert message={mutation.error.message} />}

      {mutation.isPending && (
        <div className="animate-pulse rounded bg-[var(--border)] h-48" />
      )}

      {result && !mutation.isPending && (
        <div className="space-y-4">
          <p className="text-sm text-[var(--text-muted)]">
            Population: {result.n_subjects} subjects
          </p>

          <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className={headerClass}>Parameter</th>
                  <th className={`${headerClass} text-right`}>Median</th>
                  <th className={`${headerClass} text-right`}>Mean</th>
                  <th className={`${headerClass} text-right`}>CV%</th>
                  <th className={`${headerClass} text-right`}>P5</th>
                  <th className={`${headerClass} text-right`}>P95</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-[var(--border)]">
                  <td className={`${cellClass} font-medium`}>Cmax (mg/L)</td>
                  <td className={`${cellClass} text-right`}>{formatNum(result.cmax_median)}</td>
                  <td className={`${cellClass} text-right`}>{formatNum(result.cmax_mean)}</td>
                  <td className={`${cellClass} text-right`}>{formatNum(result.cmax_cv_pct, 1)}</td>
                  <td className={`${cellClass} text-right`}>{formatNum(result.cmax_p5)}</td>
                  <td className={`${cellClass} text-right`}>{formatNum(result.cmax_p95)}</td>
                </tr>
                <tr>
                  <td className={`${cellClass} font-medium`}>AUC (mg{"\u00B7"}h/L)</td>
                  <td className={`${cellClass} text-right`}>{formatNum(result.auc_median)}</td>
                  <td className={`${cellClass} text-right`}>{formatNum(result.auc_mean)}</td>
                  <td className={`${cellClass} text-right`}>{formatNum(result.auc_cv_pct, 1)}</td>
                  <td className={`${cellClass} text-right`}>--</td>
                  <td className={`${cellClass} text-right`}>--</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}
    </PageShell>
  );
}
