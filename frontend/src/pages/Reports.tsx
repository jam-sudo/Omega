import { useState, useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import PageShell from "../components/layout/PageShell";
import SmilesInput from "../components/SmilesInput";
import ErrorAlert from "../components/ErrorAlert";
import { api } from "../api/client";
import { downloadBlob } from "../lib/utils";

export default function Reports() {
  const [smiles, setSmiles] = useState("");
  const [drugName, setDrugName] = useState("");
  const [dose, setDose] = useState("100");
  const [route, setRoute] = useState("oral");
  const iframeRef = useRef<HTMLIFrameElement>(null);

  const mutation = useMutation<string, Error>({
    mutationFn: async () => {
      const res = await api.report({
        smiles: smiles.trim(),
        drug_name: drugName || undefined,
        dose_mg: Number(dose) || 100,
        route,
      });
      // Decode base64 HTML
      return atob(res.data);
    },
  });

  const handleGenerate = () => {
    if (!smiles.trim()) return;
    mutation.mutate();
  };

  const handleDownload = () => {
    if (!mutation.data) return;
    const filename = `${drugName || "report"}_pk_report.html`;
    downloadBlob(mutation.data, filename, "text/html");
  };

  const inputClass =
    "w-full rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm text-[var(--text)] focus:outline-none focus:ring-1 focus:ring-blue-500";

  return (
    <PageShell title="Reports">
      <div className="space-y-4 mb-8">
        <SmilesInput value={smiles} onChange={setSmiles} />

        <div className="flex items-end gap-4 flex-wrap">
          <div>
            <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">
              Drug Name
            </label>
            <input
              type="text"
              value={drugName}
              onChange={(e) => setDrugName(e.target.value)}
              placeholder="e.g. Aspirin"
              className={`${inputClass} w-40`}
            />
          </div>
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
            onClick={handleGenerate}
            disabled={!smiles.trim() || mutation.isPending}
            className="px-5 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {mutation.isPending ? "Generating..." : "Generate Report"}
          </button>
        </div>
      </div>

      {mutation.error && <ErrorAlert message={mutation.error.message} />}

      {mutation.isPending && (
        <div className="animate-pulse rounded bg-[var(--border)] h-96" />
      )}

      {mutation.data && !mutation.isPending && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-[var(--text-muted)]">Report preview</p>
            <button
              onClick={handleDownload}
              className="px-4 py-2 rounded-md border border-[var(--border)] text-sm text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-[var(--border)] transition-colors"
            >
              Download HTML
            </button>
          </div>

          <iframe
            ref={iframeRef}
            srcDoc={mutation.data}
            title="PK Report"
            className="w-full rounded-lg border border-[var(--border)] bg-white"
            style={{ height: "70vh" }}
            sandbox="allow-same-origin"
          />
        </div>
      )}
    </PageShell>
  );
}
