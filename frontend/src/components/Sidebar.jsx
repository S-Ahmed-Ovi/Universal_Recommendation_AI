import { useEffect, useLayoutEffect, useRef } from "react";
import {
  Bell,
  Boxes,
  BrainCircuit,
  Building2,
  CalendarClock,
  ChartNoAxesCombined,
  Code2,
  Database,
  Gauge,
  LineChart,
  Rocket,
  Settings2,
  Trophy,
  Workflow,
} from "lucide-react";
import { NavLink, useLocation } from "react-router-dom";
import WorkspaceControls from "./WorkspaceControls";

const SIDEBAR_SCROLL_KEY = "proactive_ai_sidebar_scroll_top";

const groups = [
  {
    label: "Overview",
    items: [
      { label: "Dashboard", meta: "Workspace overview", to: "/dashboard", icon: Gauge },
      { label: "Results Flow", meta: "Post-training navigation", to: "/results", icon: ChartNoAxesCombined },
      { label: "Analytics", meta: "Operational usage insights", to: "/analytics", icon: LineChart },
      { label: "API Playground", meta: "Test live requests", to: "/api", icon: Workflow },
    ],
  },
  {
    label: "Data & Models",
    items: [
      { label: "Datasets", meta: "Prepare data", to: "/datasets", icon: Database },
      { label: "Training", meta: "File or SQL jobs", to: "/training", icon: BrainCircuit },
      { label: "Scheduler", meta: "Automated retraining", to: "/scheduler", icon: CalendarClock },
      { label: "Benchmarks", meta: "Leaderboard and metrics", to: "/benchmarks", icon: Trophy },
      { label: "Registry", meta: "Saved model inventory", to: "/registry", icon: Boxes },
      { label: "Deployment", meta: "Promote and rollback", to: "/deployment", icon: Rocket },
      { label: "Integrations", meta: "SDK and API docs", to: "/integrations", icon: Code2 },
    ],
  },
  {
    label: "Workspace",
    items: [
      { label: "Tenants", meta: "Clients and API keys", to: "/tenants", icon: Building2 },
      { label: "Notifications", meta: "Alerts and events", to: "/notifications", icon: Bell },
      { label: "Settings", meta: "Preferences and policies", to: "/settings", icon: Settings2 },
    ],
  },
];

export default function Sidebar(){
  const { pathname } = useLocation();
  const sidebarRef = useRef(null);

  function persistScrollPosition() {
    if (typeof window === "undefined" || !sidebarRef.current) {
      return;
    }
    window.sessionStorage.setItem(SIDEBAR_SCROLL_KEY, String(sidebarRef.current.scrollTop));
  }

  // Restore the sidebar's own scroll position when the layout remounts on route changes.
  useLayoutEffect(() => {
    if (typeof window === "undefined" || !sidebarRef.current) {
      return;
    }
    const stored = Number(window.sessionStorage.getItem(SIDEBAR_SCROLL_KEY));
    if (Number.isFinite(stored)) {
      sidebarRef.current.scrollTop = stored;
    }
  }, [pathname]);

  useEffect(() => {
    const sidebarNode = sidebarRef.current;
    if (!sidebarNode) {
      return undefined;
    }

    persistScrollPosition();
    sidebarNode.addEventListener("scroll", persistScrollPosition, { passive: true });

    return () => {
      persistScrollPosition();
      sidebarNode.removeEventListener("scroll", persistScrollPosition);
    };
  }, [pathname]);

  return (
    <aside className="sidebar" ref={sidebarRef}>
      <div className="brand-lockup">
        <div className="brand-mark">PA</div>
        <div>
          <p className="brand-kicker">Recommendation Platform</p>
          <h1 className="brand-title">Proactive AI</h1>

        </div>
      </div>

      <div className="nav-groups">
        {groups.map((group) => (
          <section className="nav-group" key={group.label}>
            <p className="nav-group-label">{group.label}</p>
            <div className="nav-group-items">
              {group.items.map((item) => (
                <NavLink
                  className={({ isActive }) => isActive ? "nav-item nav-item-active" : "nav-item"}
                  end
                  key={item.to}
                  onClick={persistScrollPosition}
                  preventScrollReset
                  to={item.to}
                >
                  <div className="nav-item-icon">
                    <item.icon size={18} />
                  </div>
                  <div className="nav-item-copy">
                    <span className="nav-item-title">{item.label}</span>
                    <span className="nav-item-meta">{item.meta}</span>
                  </div>
                </NavLink>
              ))}
            </div>
          </section>
        ))}
      </div>

      <WorkspaceControls title="Settings Access" />

      <div className="sidebar-footer">
        <p className="eyebrow">Operator Focus</p>
        <p>
          Use benchmarks to justify promotion, then validate the selected model path in the API playground.
        </p>
      </div>
    </aside>
  )
}
