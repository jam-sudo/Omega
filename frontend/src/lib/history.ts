const KEY = "omega_history";
const MAX = 20;

export interface HistoryEntry {
  smiles: string;
  dose: number;
  route: string;
  cmax: number;
  auc: number;
  thalf: number;
  timestamp: number;
}

export function getHistory(): HistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(KEY) || "[]");
  } catch {
    return [];
  }
}

export function addToHistory(entry: HistoryEntry): void {
  const list = getHistory();
  const filtered = list.filter((e) => e.smiles !== entry.smiles);
  filtered.unshift(entry);
  localStorage.setItem(KEY, JSON.stringify(filtered.slice(0, MAX)));
}

export function clearHistory(): void {
  localStorage.removeItem(KEY);
}
