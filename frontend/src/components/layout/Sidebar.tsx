import { NavLink } from "react-router-dom";
import { useState } from "react";
import {
  LayoutDashboard,
  FlaskConical,
  GitCompareArrows,
  Pill,
  AlertTriangle,
  Users,
  FileText,
  PanelLeftClose,
  PanelLeftOpen,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";

interface NavItem {
  path: string;
  label: string;
  icon: LucideIcon;
}

const NAV_ITEMS: NavItem[] = [
  { path: "/", label: "Dashboard", icon: LayoutDashboard },
  { path: "/predict", label: "Predict", icon: FlaskConical },
  { path: "/compare", label: "Compare", icon: GitCompareArrows },
  { path: "/dose-optimize", label: "Dose Opt", icon: Pill },
  { path: "/ddi", label: "DDI", icon: AlertTriangle },
  { path: "/population", label: "PopPK", icon: Users },
  { path: "/reports", label: "Reports", icon: FileText },
];

export default function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={`h-screen sticky top-0 flex flex-col border-r transition-all duration-200 shrink-0 ${
        collapsed ? "w-[60px]" : "w-[240px]"
      }`}
      style={{ background: "var(--surface)" }}
    >
      {/* Logo */}
      <div className="h-14 flex items-center gap-2.5 px-4 border-b border-[var(--border)]">
        <span className="text-[#fafafa] font-light text-xl">Ω</span>
        {!collapsed && (
          <span className="font-light text-sm text-[var(--text)]">Omega PBPK</span>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 flex flex-col gap-0.5 px-2 py-3">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === "/"}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors ${
                  isActive
                    ? "bg-white/[0.04] text-[#fafafa] font-medium border-l-2 border-[#3b82f6]"
                    : "text-[var(--text-muted)] hover:text-[var(--text)] hover:bg-white/[0.04]"
                }`
              }
              title={collapsed ? item.label : undefined}
            >
              <Icon size={18} strokeWidth={1.75} />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          );
        })}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="h-10 flex items-center justify-center gap-2 border-t border-[var(--border)] text-[var(--text-muted)] hover:text-[var(--text)] transition-colors text-xs"
      >
        {collapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
        {!collapsed && <span>Collapse</span>}
      </button>
    </aside>
  );
}
