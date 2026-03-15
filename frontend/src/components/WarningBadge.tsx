interface Props {
  warnings: string[];
  riskLevel?: string;
}

function riskColor(level: string): string {
  switch (level) {
    case "high": return "border-red-500/30 bg-red-500/10 text-red-400";
    case "medium": return "border-yellow-500/30 bg-yellow-500/10 text-yellow-400";
    case "low": return "border-green-500/30 bg-green-500/10 text-green-400";
    default: return "border-[var(--border)] bg-[var(--surface)] text-[var(--text-muted)]";
  }
}

export default function WarningBadge({ warnings, riskLevel }: Props) {
  if (warnings.length === 0) return null;

  return (
    <div className="space-y-2">
      {riskLevel && (
        <span className={`inline-block text-xs font-medium px-2 py-0.5 rounded border ${riskColor(riskLevel)}`}>
          Risk: {riskLevel}
        </span>
      )}
      <ul className="space-y-1">
        {warnings.map((w, i) => (
          <li
            key={i}
            className="text-sm text-yellow-400/90 flex items-start gap-2"
          >
            <span className="mt-0.5 flex-shrink-0">!</span>
            <span>{w}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
