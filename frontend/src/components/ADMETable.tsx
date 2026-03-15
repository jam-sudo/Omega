import { formatNum } from "../lib/utils";

interface Props {
  adme: Record<string, number | string>;
}

const DISPLAY_ROWS: { key: string; label: string; unit: string }[] = [
  { key: "mw", label: "MW", unit: "g/mol" },
  { key: "logP", label: "logP", unit: "" },
  { key: "logS", label: "logS", unit: "" },
  { key: "fup", label: "fup", unit: "" },
  { key: "rbp", label: "RBP", unit: "" },
  { key: "peff", label: "Peff", unit: "10\u207B\u2074 cm/s" },
  { key: "clint_3a4", label: "CLint 3A4", unit: "\u00B5L/min/pmol" },
  { key: "herg_ic50_uM", label: "hERG IC50", unit: "\u00B5M" },
];

export default function ADMETable({ adme }: Props) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
      <div className="px-4 py-3 border-b border-[var(--border)]">
        <h3 className="text-sm font-medium text-[var(--text)]">ADME Properties</h3>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[var(--border)] text-[var(--text-muted)]">
            <th className="text-left px-4 py-2 font-medium">Property</th>
            <th className="text-right px-4 py-2 font-medium">Value</th>
            <th className="text-right px-4 py-2 font-medium">CI Low</th>
            <th className="text-right px-4 py-2 font-medium">CI High</th>
          </tr>
        </thead>
        <tbody>
          {DISPLAY_ROWS.map((row) => {
            const val = adme[row.key];
            const lo = adme[`${row.key}_lo`];
            const hi = adme[`${row.key}_hi`];
            if (val == null) return null;
            return (
              <tr key={row.key} className="border-b border-[var(--border)] last:border-b-0">
                <td className="px-4 py-2 text-[var(--text)]">
                  {row.label}
                  {row.unit && (
                    <span className="text-[var(--text-muted)] ml-1 text-xs">({row.unit})</span>
                  )}
                </td>
                <td className="px-4 py-2 text-right font-mono text-[var(--text)]">
                  {typeof val === "number" ? formatNum(val) : val}
                </td>
                <td className="px-4 py-2 text-right font-mono text-[var(--text-muted)]">
                  {typeof lo === "number" ? formatNum(lo) : "\u2014"}
                </td>
                <td className="px-4 py-2 text-right font-mono text-[var(--text-muted)]">
                  {typeof hi === "number" ? formatNum(hi) : "\u2014"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
