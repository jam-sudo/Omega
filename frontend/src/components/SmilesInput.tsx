import { useEffect, useRef, useState } from "react";
import { ChevronDown } from "lucide-react";

interface Props {
  value: string;
  onChange: (v: string) => void;
  showPreview?: boolean;
}

const EXAMPLES = [
  { name: "Caffeine", smiles: "Cn1c(=O)c2c(ncn2C)n(C)c1=O" },
  { name: "Ibuprofen", smiles: "CC(C)Cc1ccc(C(C)C(=O)O)cc1" },
  { name: "Acetaminophen", smiles: "CC(=O)Nc1ccc(O)cc1" },
  { name: "Metoprolol", smiles: "COCCc1ccc(OCC(O)CNC(C)C)cc1" },
  { name: "Warfarin", smiles: "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O" },
  { name: "Diazepam", smiles: "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21" },
];

export default function SmilesInput({ value, onChange, showPreview = true }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [showExamples, setShowExamples] = useState(false);
  const [svgError, setSvgError] = useState(false);

  useEffect(() => {
    if (!showPreview || !svgRef.current || !value.trim()) {
      if (svgRef.current) svgRef.current.innerHTML = "";
      setSvgError(false);
      return;
    }

    let cancelled = false;

    (async () => {
      try {
        const SmilesDrawer = (await import("smiles-drawer")).default;
        if (cancelled || !svgRef.current) return;

        const svg = svgRef.current;
        const w = svg.clientWidth || 600;
        const h = 220;

        const drawer = new SmilesDrawer.SvgDrawer({
          width: w,
          height: h,
          bondThickness: 1.2,
          fontSizeLarge: 12,
          fontSizeSmall: 8,
          padding: 30,
          compactDrawing: false,
          themes: {
            dark: {
              C: "#d4d4d8",
              O: "#f87171",
              N: "#60a5fa",
              S: "#facc15",
              F: "#34d399",
              Cl: "#34d399",
              Br: "#a78bfa",
              P: "#fb923c",
              H: "#71717a",
              BACKGROUND: "transparent",
            },
          },
        });

        SmilesDrawer.parse(
          value,
          (tree: unknown) => {
            if (cancelled || !svgRef.current) return;
            try {
              drawer.draw(tree, svg, "dark");
              setSvgError(false);
            } catch {
              svg.innerHTML = "";
              setSvgError(true);
            }
          },
          () => {
            // Parse error
            if (!cancelled && svgRef.current) {
              svgRef.current.innerHTML = "";
              setSvgError(true);
            }
          },
        );
      } catch {
        setSvgError(true);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [value, showPreview]);

  return (
    <div className="space-y-3">
      <div className="relative">
        <label className="block text-sm font-medium text-[var(--text-muted)] mb-1">SMILES</label>
        <div className="flex gap-1">
          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="Enter SMILES or select an example →"
            className="flex-1 rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm font-mono text-[var(--text)] placeholder:text-[var(--text-muted)]/50 focus:outline-none focus:ring-1 focus:ring-blue-500"
            spellCheck={false}
            autoComplete="off"
          />
          <div className="relative">
            <button
              onClick={() => setShowExamples(!showExamples)}
              className="h-full px-2 rounded-md border border-[var(--border)] bg-[var(--surface)] text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-white/5 transition-colors"
              title="Example drugs"
            >
              <ChevronDown size={14} />
            </button>
            {showExamples && (
              <div className="absolute right-0 top-full mt-1 w-52 rounded-lg border border-[var(--border)] bg-[#1a1a1a] shadow-xl z-50 py-1">
                {EXAMPLES.map((ex) => (
                  <button
                    key={ex.name}
                    onClick={() => {
                      onChange(ex.smiles);
                      setShowExamples(false);
                    }}
                    className="w-full text-left px-3 py-1.5 text-sm hover:bg-white/5 transition-colors flex justify-between items-center"
                  >
                    <span className="text-[var(--text)]">{ex.name}</span>
                    <span className="text-[10px] text-[var(--text-muted)] font-mono truncate max-w-[80px]">
                      {ex.smiles}
                    </span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
      {showPreview && value.trim() && (
        <div className="rounded-lg border border-[var(--border)] overflow-hidden" style={{ background: "#141414" }}>
          <svg
            ref={svgRef}
            className="w-full"
            style={{ height: 220, display: svgError ? "none" : "block" }}
          />
          {svgError && (
            <div className="flex items-center justify-center h-[220px] text-sm text-[var(--text-muted)]">
              Invalid SMILES — cannot render structure
            </div>
          )}
        </div>
      )}
    </div>
  );
}
