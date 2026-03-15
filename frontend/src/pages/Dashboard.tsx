import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { FlaskConical, ArrowRight, Trash2 } from "lucide-react";
import PageShell from "../components/layout/PageShell";
import SmilesInput from "../components/SmilesInput";
import { getHistory, clearHistory, type HistoryEntry } from "../lib/history";
import { formatNum } from "../lib/utils";

const EXAMPLE_DRUGS = [
  { name: "Caffeine", smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O", dose: 200 },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(C(C)C(=O)O)cc1", dose: 400 },
  { name: "Aspirin", smiles: "CC(=O)Oc1ccccc1C(=O)O", dose: 500 },
  { name: "Metoprolol", smiles: "COCCc1ccc(OCC(O)CNC(C)C)cc1", dose: 100 },
];

export default function Dashboard() {
  const navigate = useNavigate();
  const [smiles, setSmiles] = useState("");
  const [history, setHistory] = useState<HistoryEntry[]>(getHistory());

  const handleGo = () => {
    if (smiles.trim()) navigate(`/predict?smiles=${encodeURIComponent(smiles)}`);
  };

  const handleExample = (drug: (typeof EXAMPLE_DRUGS)[number]) => {
    navigate(`/predict?smiles=${encodeURIComponent(drug.smiles)}&dose=${drug.dose}`);
  };

  return (
    <PageShell title="Dashboard">
      {/* Quick Predict */}
      <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5 mb-8">
        <h2 className="text-sm font-medium text-[var(--text-muted)] uppercase tracking-wide mb-3">
          Quick Predict
        </h2>
        <div className="flex gap-3 items-end">
          <div className="flex-1">
            <SmilesInput value={smiles} onChange={setSmiles} showPreview={false} />
          </div>
          <button
            onClick={handleGo}
            disabled={!smiles.trim()}
            className="px-5 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
          >
            Predict <ArrowRight size={14} />
          </button>
        </div>
      </div>

      {/* Recent Predictions or Empty State */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-medium text-[var(--text-muted)] uppercase tracking-wide">
            Recent Predictions
          </h2>
          {history.length > 0 && (
            <button
              onClick={() => {
                clearHistory();
                setHistory([]);
              }}
              className="text-xs text-[var(--text-muted)] hover:text-red-400 flex items-center gap-1 transition-colors"
            >
              <Trash2 size={12} /> Clear
            </button>
          )}
        </div>

        {history.length === 0 ? (
          /* Empty state with example drugs */
          <div className="rounded-lg border border-dashed border-[var(--border)] p-8 text-center">
            <FlaskConical
              size={40}
              className="mx-auto mb-3 text-[var(--text-muted)]"
              strokeWidth={1.25}
            />
            <p className="text-sm text-[var(--text-muted)] mb-4">
              No predictions yet. Try one of these examples:
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {EXAMPLE_DRUGS.map((drug) => (
                <button
                  key={drug.name}
                  onClick={() => handleExample(drug)}
                  className="px-3 py-1.5 rounded-md border border-[var(--border)] bg-[var(--surface)] text-sm text-[var(--text)] hover:bg-white/5 hover:border-blue-500/30 transition-colors"
                >
                  {drug.name}
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* History table */
          <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--border)] text-[var(--text-muted)] text-xs uppercase tracking-wide">
                  <th className="text-left px-4 py-2.5">SMILES</th>
                  <th className="text-right px-4 py-2.5">Cmax</th>
                  <th className="text-right px-4 py-2.5">AUC</th>
                  <th className="text-right px-4 py-2.5">t½</th>
                  <th className="px-4 py-2.5 w-8" />
                </tr>
              </thead>
              <tbody>
                {history.map((h, i) => (
                  <tr
                    key={i}
                    className="border-b border-[var(--border)] last:border-0 hover:bg-white/[0.03] cursor-pointer transition-colors"
                    onClick={() =>
                      navigate(
                        `/predict?smiles=${encodeURIComponent(h.smiles)}&dose=${h.dose}&route=${h.route}`,
                      )
                    }
                  >
                    <td className="px-4 py-2.5 font-mono text-xs max-w-[300px] truncate text-[var(--text)]">
                      {h.smiles}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-[var(--text)]">
                      {formatNum(h.cmax)}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-[var(--text)]">
                      {formatNum(h.auc)}
                    </td>
                    <td className="px-4 py-2.5 text-right tabular-nums text-[var(--text)]">
                      {formatNum(h.thalf)}h
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <ArrowRight size={14} className="text-[var(--text-muted)]" />
                    </td>
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
