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
import type { DDIResponse } from "../api/types";

export default function DDIChecker() {
  // Victim
  const [smiles, setSmiles] = useState("");
  const [dose, setDose] = useState("100");

  // Perpetrator
  const [perpName, setPerpName] = useState("");
  const [cypTarget, setCypTarget] = useState("CYP3A4");
  const [ki, setKi] = useState("");
  const [mechanism, setMechanism] = useState("competitive");
  const [perpCmax, setPerpCmax] = useState("");

  const mutation = useMutation<DDIResponse, Error>({
    mutationFn: async () => {
      // Step 1: predict victim ADME
      const adme = await api.predict({
        smiles: smiles.trim(),
        dose_mg: Number(dose) || 100,
      });
      const victimDrug = admeToDrugRequest(adme, Number(dose) || 100);

      // Step 2: call DDI simulate
      return api.ddiSimulate({
        victim_drug: victimDrug,
        perpetrator_name: perpName || "Inhibitor",
        perpetrator_ki_uM: Number(ki) || 1,
        perpetrator_target_enzyme: cypTarget,
        perpetrator_mechanism: mechanism,
        perpetrator_cmax_uM: perpCmax ? Number(perpCmax) : undefined,
      });
    },
  });

  const handleSimulate = () => {
    if (!smiles.trim() || !ki) return;
    mutation.mutate();
  };

  const result = mutation.data;

  const datasets: Dataset[] | null = result
    ? [
        { time: result.time_h, conc: result.cp_alone, label: "Alone", color: "#3b82f6" },
        { time: result.time_h, conc: result.cp_with_inhibitor, label: "With Inhibitor", color: "#ef4444" },
      ]
    : null;

  const inputClass =
    "w-full rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] focus:outline-none focus:ring-1 focus:ring-blue-500";
  const selectClass = inputClass;

  const magnitudeColor = (mag: string) => {
    switch (mag.toLowerCase()) {
      case "strong": return "text-red-400";
      case "moderate": return "text-yellow-400";
      case "weak": return "text-green-400";
      default: return "text-[var(--text-muted)]";
    }
  };

  return (
    <PageShell title="DDI Checker">
      <div className="space-y-6 mb-8">
        {/* Victim section */}
        <div>
          <h2 className="text-sm font-medium text-[var(--text-muted)] mb-3 uppercase tracking-wider">
            Victim Drug
          </h2>
          <SmilesInput value={smiles} onChange={setSmiles} />
          <div className="mt-3">
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
        </div>

        {/* Perpetrator section */}
        <div>
          <h2 className="text-sm font-medium text-[var(--text-muted)] mb-3 uppercase tracking-wider">
            Perpetrator (Inhibitor)
          </h2>
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <div>
              <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">Name</label>
              <input
                type="text"
                value={perpName}
                onChange={(e) => setPerpName(e.target.value)}
                placeholder="e.g. Ketoconazole"
                className={inputClass}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">CYP Target</label>
              <select value={cypTarget} onChange={(e) => setCypTarget(e.target.value)} className={selectClass}>
                <option value="CYP3A4">CYP3A4</option>
                <option value="CYP2D6">CYP2D6</option>
                <option value="CYP2C9">CYP2C9</option>
                <option value="CYP2C19">CYP2C19</option>
                <option value="CYP1A2">CYP1A2</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">
                Ki (&micro;M)
              </label>
              <input
                type="number"
                value={ki}
                onChange={(e) => setKi(e.target.value)}
                min={0}
                step={0.01}
                placeholder="e.g. 0.03"
                className={inputClass}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">Mechanism</label>
              <select value={mechanism} onChange={(e) => setMechanism(e.target.value)} className={selectClass}>
                <option value="competitive">Competitive</option>
                <option value="noncompetitive">Non-competitive</option>
                <option value="mechanism_based">Mechanism-based</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">
                Cmax (&micro;M) <span className="text-[var(--text-muted)] font-normal">(opt.)</span>
              </label>
              <input
                type="number"
                value={perpCmax}
                onChange={(e) => setPerpCmax(e.target.value)}
                min={0}
                step={0.1}
                placeholder="optional"
                className={inputClass}
              />
            </div>
          </div>
        </div>

        <button
          onClick={handleSimulate}
          disabled={!smiles.trim() || !ki || mutation.isPending}
          className="px-5 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {mutation.isPending ? "Simulating..." : "Simulate DDI"}
        </button>
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
          {/* Summary cards */}
          <div className="grid grid-cols-3 gap-4">
            <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
              <p className="text-xs text-[var(--text-muted)] mb-1">AUC Ratio</p>
              <p className="text-xl font-semibold text-[var(--text)]">{formatNum(result.auc_ratio)}</p>
            </div>
            <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
              <p className="text-xs text-[var(--text-muted)] mb-1">Cmax Ratio</p>
              <p className="text-xl font-semibold text-[var(--text)]">{formatNum(result.cmax_ratio)}</p>
            </div>
            <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
              <p className="text-xs text-[var(--text-muted)] mb-1">Interaction</p>
              <p className={`text-xl font-semibold ${magnitudeColor(result.interaction_magnitude)}`}>
                {result.interaction_magnitude}
              </p>
            </div>
          </div>

          {/* C(t) overlay */}
          {datasets && <PKCurve datasets={datasets} />}
        </div>
      )}
    </PageShell>
  );
}
