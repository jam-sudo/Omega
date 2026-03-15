import { NavLink } from "react-router-dom";
import { useState } from "react";

const NAV_ITEMS = [
  { path: "/", label: "Dashboard", icon: "\u{1F3E0}" },
  { path: "/predict", label: "Predict", icon: "\u{1F52C}" },
  { path: "/compare", label: "Compare", icon: "\u2696\uFE0F" },
  { path: "/dose-optimize", label: "Dose Opt", icon: "\u{1F48A}" },
  { path: "/ddi", label: "DDI", icon: "\u26A0\uFE0F" },
  { path: "/population", label: "PopPK", icon: "\u{1F465}" },
  { path: "/reports", label: "Reports", icon: "\u{1F4C4}" },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={`h-screen sticky top-0 flex flex-col border-r transition-all duration-200 ${
        collapsed ? "w-[60px]" : "w-[240px]"
      }`}
      style={{ background: "var(--surface)" }}
    >
      <div className="p-4 font-bold text-lg flex items-center gap-2">
        <span className="text-xl">{"\u03A9"}</span>
        {!collapsed && <span>Omega PBPK</span>}
      </div>
      <nav className="flex-1 flex flex-col gap-1 px-2">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === "/"}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${
                isActive
                  ? "bg-blue-500/10 text-blue-400"
                  : "text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-white/5"
              }`
            }
          >
            <span>{item.icon}</span>
            {!collapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="p-3 text-[var(--text-muted)] hover:text-[var(--text)] text-xs"
      >
        {collapsed ? "\u2192" : "\u2190 Collapse"}
      </button>
    </aside>
  );
}
