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
    if (smiles.trim()) navigate(`/predict?smiles=${encodeURIComponent(smiles)}`);
  };

  return (
    <PageShell title="Dashboard">
      <div className="mb-8">
        <h2 className="text-sm font-medium text-[var(--text-muted)] uppercase tracking-wide mb-2">Quick Predict</h2>
        <div className="flex gap-3 items-end">
          <div className="flex-1"><SmilesInput value={smiles} onChange={setSmiles} showPreview={false} /></div>
          <button onClick={handleGo} disabled={!smiles.trim()}
            className="px-6 py-2 rounded-md bg-blue-600 text-white text-sm font-medium hover:bg-blue-500 disabled:opacity-50">Go →</button>
        </div>
      </div>
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-medium text-[var(--text-muted)] uppercase tracking-wide">Recent Predictions</h2>
          {history.length > 0 && (
            <button onClick={() => { clearHistory(); setHistory([]); }} className="text-xs text-[var(--text-muted)] hover:text-red-400">Clear</button>
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
