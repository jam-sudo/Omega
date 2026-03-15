export function formatNum(n: number, decimals = 2): string {
  if (n === 0) return "0";
  if (Math.abs(n) < 0.001) return n.toExponential(1);
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  return n.toFixed(decimals);
}

export function confidenceColor(level: string): string {
  switch (level) {
    case "high": return "var(--success)";
    case "medium": return "var(--warning)";
    case "low": return "var(--danger)";
    default: return "var(--text-muted)";
  }
}

export function downloadBlob(data: string, filename: string, mime: string): void {
  const blob = new Blob([data], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function exportCSV(timeH: number[], cpMgL: number[], filename: string): void {
  const header = "time_h,cp_mg_L\n";
  const rows = timeH.map((t, i) => `${t},${cpMgL[i]}`).join("\n");
  downloadBlob(header + rows, filename, "text/csv");
}
