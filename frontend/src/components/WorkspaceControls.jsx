import { LogOut, MoonStar, SunMedium } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { clearAuthToken, clearTenantContext } from "../auth";
import { useTheme } from "../theme";

export default function WorkspaceControls({ compact = false, title = "Settings Access" }) {
  const navigate = useNavigate();
  const { theme, toggleTheme } = useTheme();

  function handleLogout() {
    clearAuthToken();
    clearTenantContext();
    navigate("/", { replace: true });
  }

  return (
    <div className={`sidebar-controls${compact ? " sidebar-controls-compact" : ""}`}>
      <div className="sidebar-controls-header">
        <p className="eyebrow">{title}</p>
        <span className="status-pill status-muted">Workspace Actions</span>
      </div>

      <button className="sidebar-control-button" onClick={toggleTheme} type="button">
        <span className="sidebar-control-icon">
          {theme === "dark" ? <SunMedium size={18} /> : <MoonStar size={18} />}
        </span>
        <span className="sidebar-control-copy">
          <strong>{theme === "dark" ? "Light Mode" : "Dark Mode"}</strong>
          <small>Switch workspace appearance</small>
        </span>
      </button>

      <button className="sidebar-control-button sidebar-control-danger" onClick={handleLogout} type="button">
        <span className="sidebar-control-icon">
          <LogOut size={18} />
        </span>
        <span className="sidebar-control-copy">
          <strong>Log Out</strong>
          <small>End the current admin session</small>
        </span>
      </button>
    </div>
  );
}
