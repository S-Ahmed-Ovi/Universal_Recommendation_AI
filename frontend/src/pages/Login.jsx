import { useCallback, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { publicRequest } from "../api";
import {
  clearTenantContext,
  setAuthToken,
  setTenantContext,
} from "../auth";
import { useAsyncData } from "../hooks/useAsyncData";
import { useTheme } from "../theme";

const launchPoints = [
  "Live tenant key management",
  "Asynchronous model training orchestration",
  "Recommendation API testing with production auth",
];

const DEFAULT_REGISTER_META = Object.freeze({
  assignable_on_registration: ["USER"],
  default_role: "USER",
  roles: [],
});

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const USERNAME_RE = /^[a-z0-9][a-z0-9_.-]{2,62}$/;

function extractMessage(detail, fallback) {
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  if (Array.isArray(detail) && detail.length) {
    return detail
      .map((item) => {
        const path = Array.isArray(item?.loc) ? item.loc[item.loc.length - 1] : "";
        const prefix = path ? `${path}: ` : "";
        return `${prefix}${item?.msg || "Invalid value"}`;
      })
      .join(" ");
  }

  if (detail && typeof detail === "object") {
    if (typeof detail.message === "string" && detail.message.trim()) {
      return detail.message;
    }
    if (typeof detail.detail === "string" && detail.detail.trim()) {
      return detail.detail;
    }
  }

  return fallback;
}

function fieldErrorsFromDetail(detail) {
  const next = {};

  if (!Array.isArray(detail)) {
    return next;
  }

  for (const item of detail) {
    const key = Array.isArray(item?.loc) ? item.loc[item.loc.length - 1] : "";
    if (!key) {
      continue;
    }
    if (key === "confirm_password") {
      next.confirmPassword = item?.msg || "Passwords must match.";
    } else if (key === "email") {
      next.email = item?.msg || "Email is invalid.";
    } else if (key === "username") {
      next.username = item?.msg || "Username is invalid.";
    } else if (key === "password") {
      next.password = item?.msg || "Password is invalid.";
    }
  }

  return next;
}

export default function Login() {
  const navigate = useNavigate();
  const { theme, toggleTheme } = useTheme();
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [fieldErrors, setFieldErrors] = useState({
    confirmPassword: "",
    email: "",
    password: "",
    username: "",
  });
  const [loading, setLoading] = useState(false);
  const showDevHint = useMemo(() => Boolean(import.meta.env.DEV), []);
  const isRegistering = mode === "register";

  const loadRegisterMeta = useCallback(({ signal }) => (
    publicRequest("/admin/auth/register-meta", { signal })
  ), []);
  const { data: registerMeta } = useAsyncData(loadRegisterMeta, {
    initialData: DEFAULT_REGISTER_META,
  });

  const roleLabel = useMemo(() => {
    const defaultRole = registerMeta?.default_role || "USER";
    const catalog = registerMeta?.roles || [];
    const matched = catalog.find((role) => role.role === defaultRole);
    return matched?.label || defaultRole;
  }, [registerMeta]);

  function validateForm() {
    const nextErrors = {
      confirmPassword: "",
      email: "",
      password: "",
      username: "",
    };
    const trimmedUsername = username.trim();
    const trimmedEmail = email.trim().toLowerCase();

    if (!trimmedUsername) {
      nextErrors.username = "Username is required.";
    } else if (!USERNAME_RE.test(trimmedUsername)) {
      nextErrors.username = "Username must be 3-63 chars and use lowercase letters, digits, dots, underscores, or hyphens.";
    }

    if (isRegistering) {
      if (!trimmedEmail) {
        nextErrors.email = "Email is required.";
      } else if (!EMAIL_RE.test(trimmedEmail)) {
        nextErrors.email = "Enter a valid email address.";
      }

      if (!password) {
        nextErrors.password = "Password is required.";
      } else if (password.length < 8) {
        nextErrors.password = "Password must be at least 8 characters.";
      }

      if (!confirmPassword) {
        nextErrors.confirmPassword = "Confirm your password.";
      } else if (password !== confirmPassword) {
        nextErrors.confirmPassword = "Passwords do not match.";
      }
    } else if (!password) {
      nextErrors.password = "Password is required.";
    }

    setFieldErrors(nextErrors);
    return !Object.values(nextErrors).some(Boolean);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");

    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      const payload = isRegistering
        ? {
            confirm_password: confirmPassword,
            email: email.trim().toLowerCase(),
            password,
            username: username.trim(),
          }
        : {
            password,
            username: username.trim(),
          };

      const response = await publicRequest(`/admin/auth/${isRegistering ? "register" : "login"}`, {
        body: payload,
        method: "POST",
      });

      if (!response?.access_token) {
        throw new Error("Authentication response did not include an access token.");
      }

      setAuthToken(response.access_token);

      if (response.tenant_id && response.api_key) {
        setTenantContext({
          apiKey: response.api_key,
          tenantId: response.tenant_id,
        });
      } else if (response.tenant_id) {
        setTenantContext({
          apiKey: "",
          tenantId: response.tenant_id,
        });
      } else {
        clearTenantContext();
      }

      navigate(response.requires_tenant ? "/tenants" : "/dashboard", { replace: true });
    } catch (submitError) {
      const payload = submitError?.payload?.detail;
      const nextFieldErrors = fieldErrorsFromDetail(payload);
      if (Object.keys(nextFieldErrors).length) {
        setFieldErrors((current) => ({ ...current, ...nextFieldErrors }));
      }
      setError(extractMessage(payload, submitError.message || (isRegistering ? "Unable to create account." : "Unable to login.")));
    } finally {
      setLoading(false);
    }
  }

  function switchMode() {
    setMode(isRegistering ? "login" : "register");
    setError("");
    setFieldErrors({
      confirmPassword: "",
      email: "",
      password: "",
      username: "",
    });
    setConfirmPassword("");
  }

  return (
    <div className="login">
      <div className="login-stage">
        <section className="login-showcase">
          <div className="login-theme-bar">
            <button className="btn btn-secondary theme-toggle" onClick={toggleTheme} type="button">
              {theme === "dark" ? "Light Mode" : "Dark Mode"}
            </button>
          </div>
          <p className="eyebrow">Recommendation Platform</p>
          <h1>Proactive AI</h1>
          <p className="login-lead">
            A deliberate control plane for enterprise recommendation teams:
            training, registry, tenants, and live APIs in one workspace.
          </p>

          <div className="login-feature-grid">
            {launchPoints.map((point) => (
              <div className="login-feature-card" key={point}>
                <span className="status-dot" />
                <strong>{point}</strong>
              </div>
            ))}
          </div>
        </section>

        <form className="login-card" onSubmit={handleSubmit}>
          <p className="eyebrow">Operator Access</p>
          <h2>{isRegistering ? "Create your control-plane account" : "Sign in to the control plane"}</h2>
          <p className="subtle">
            {isRegistering
              ? "Register first, then create one tenant workspace for model and API operations."
              : "Use your account to enter the workspace, then attach a tenant key for live model and API operations."}
          </p>

          <div className="login-form-fields">
            <div className="field">
              <label htmlFor="login-username">Username</label>
              <input
                autoComplete="username"
                id="login-username"
                onChange={(event) => {
                  setUsername(event.target.value);
                  if (fieldErrors.username) {
                    setFieldErrors((current) => ({ ...current, username: "" }));
                  }
                }}
                placeholder="Enter your username"
                required
                type="text"
                value={username}
              />
              {fieldErrors.username ? <p className="field-error">{fieldErrors.username}</p> : null}
            </div>

            {isRegistering ? (
              <div className="field">
                <label htmlFor="login-email">Email</label>
                <input
                  autoComplete="email"
                  id="login-email"
                  onChange={(event) => {
                    setEmail(event.target.value);
                    if (fieldErrors.email) {
                      setFieldErrors((current) => ({ ...current, email: "" }));
                    }
                  }}
                  placeholder="name@company.com"
                  required
                  type="email"
                  value={email}
                />
                {fieldErrors.email ? <p className="field-error">{fieldErrors.email}</p> : null}
              </div>
            ) : null}

            <div className="field">
              <label htmlFor="login-password">Password</label>
              <input
                autoComplete={isRegistering ? "new-password" : "current-password"}
                id="login-password"
                onChange={(event) => {
                  setPassword(event.target.value);
                  if (fieldErrors.password) {
                    setFieldErrors((current) => ({ ...current, password: "" }));
                  }
                }}
                placeholder="Enter your password"
                required
                type="password"
                value={password}
              />
              {fieldErrors.password ? <p className="field-error">{fieldErrors.password}</p> : null}
            </div>

            {isRegistering ? (
              <div className="field">
                <label htmlFor="login-confirm-password">Confirm Password</label>
                <input
                  autoComplete="new-password"
                  id="login-confirm-password"
                  onChange={(event) => {
                    setConfirmPassword(event.target.value);
                    if (fieldErrors.confirmPassword) {
                      setFieldErrors((current) => ({ ...current, confirmPassword: "" }));
                    }
                  }}
                  placeholder="Re-enter your password"
                  required
                  type="password"
                  value={confirmPassword}
                />
                {fieldErrors.confirmPassword ? <p className="field-error">{fieldErrors.confirmPassword}</p> : null}
              </div>
            ) : null}
          </div>

          {error ? <div className="form-error">{error}</div> : null}

          <div className="actions-row">
            <button aria-busy={loading} className="btn" disabled={loading} type="submit">
              {loading
                ? (isRegistering ? "Creating Account..." : "Signing In...")
                : (isRegistering ? "Create Account" : "Enter Control Plane")}
            </button>
            <button
              className="btn btn-secondary"
              disabled={loading}
              onClick={switchMode}
              type="button"
            >
              {isRegistering ? "Use Existing Account" : "Create Account"}
            </button>
          </div>

          
        </form>
      </div>
    </div>
  );
}
