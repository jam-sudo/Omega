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
    queries: inputs.map((inp, _idx) => ({
      queryKey: ["predict", inp.smiles, inp.dose, "oral"] as const,
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        api.predict({ smiles: inp.smiles, dose_mg: inp.dose, route: "oral" }, signal),
      enabled: submitted && inp.smiles.trim().length > 0,
    })),
  });

  const addDrug = () => { if (inputs.length < MAX_DRUGS) setInputs([...inputs, { smiles: "", dose: 100 }]); };
  const removeDrug = (idx: number) => { setInputs(inputs.filter((_, i) => i !== idx)); setSubmitted(false); };
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
            <div className="flex-1"><SmilesInput value={inp.smiles} onChange={(v) => updateInput(idx, "smiles", v)} showPreview={false} /></div>
            <div>
              <label className="block text-xs text-[var(--text-muted)] mb-1">Dose</label>
              <input type="number" value={inp.dose} onChange={(e) => updateInput(idx, "dose", Number(e.target.value))}
                className="w-20 px-2 py-2 rounded-md border bg-[var(--surface)] text-sm text-[var(--text)]" />
            </div>
            {inputs.length > 1 && <button onClick={() => removeDrug(idx)} className="text-red-400 text-sm px-2 py-2">✕</button>}
          </div>
        ))}
      </div>
      <div className="flex gap-3 mb-6">
        {inputs.length < MAX_DRUGS && <button onClick={addDrug} className="text-sm text-blue-400 hover:text-blue-300">+ Add Drug</button>}
        <button onClick={() => setSubmitted(true)} disabled={isLoading || inputs.every((i) => !i.smiles.trim())}
          className="px-6 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-50">
          {isLoading ? "Comparing..." : "Compare"}
        </button>
      </div>
      {errors.length > 0 && errors.map((e, i) => <ErrorAlert key={i} message={e} />)}
      {results.length > 0 && (
        <div className="space-y-6">
          <PKCurve datasets={results.map((r, idx) => ({
            time: r!.time_h, conc: r!.cp_mg_L,
            label: r!.drug_name || inputs[idx]?.smiles.slice(0, 15) || `Drug ${idx + 1}`,
            color: COLORS[idx],
          }))} />
          <div className="border rounded-lg overflow-hidden" style={{ background: "var(--surface)" }}>
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b text-[var(--text-muted)] text-xs uppercase tracking-wide">
                  <th className="text-left px-4 py-2">Metric</th>
                  {results.map((r, idx) => (
                    <th key={idx} className="text-right px-4 py-2" style={{ color: COLORS[idx] }}>{r!.drug_name || `Drug ${idx + 1}`}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[{ label: "Cmax (mg/L)", key: "cmax_mg_L" }, { label: "AUC (mg·h/L)", key: "auc0t_mg_h_L" },
                  { label: "t½ (h)", key: "t_half_h" }, { label: "Tmax (h)", key: "tmax_h" }].map((row) => (
                  <tr key={row.key} className="border-b last:border-0">
                    <td className="px-4 py-2 text-[var(--text-muted)]">{row.label}</td>
                    {results.map((r, idx) => (
                      <td key={idx} className="px-4 py-2 text-right tabular-nums font-medium">
                        {formatNum((r as unknown as Record<string, number>)[row.key])}
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
