const AUTH_TOKEN_KEY = "neurorec_admin_token";
const STORAGE_LISTENER_KEY = "__neurorec_auth_storage_listener__";
const TENANT_API_KEY = "neurorec_tenant_api_key";
const TENANT_ID_KEY = "neurorec_tenant_id";
const emptySnapshot = Object.freeze({
  authToken: null,
  hasTenantApiKey: false,
  isAuthenticated: false,
  tenantApiKey: null,
  tenantId: "",
});
const listeners = new Set();

function defaultApiBaseUrl() {
  if (typeof window === "undefined") {
    return "http://127.0.0.1:8000";
  }

  return window.location.port === "5173"
    ? "http://127.0.0.1:8000"
    : window.location.origin;
}

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || defaultApiBaseUrl();

function readStorage(key) {
  if (typeof window === "undefined") {
    return null;
  }

  return window.localStorage.getItem(key);
}

function writeStorage(key, value) {
  if (typeof window === "undefined") {
    return;
  }

  if (value) {
    window.localStorage.setItem(key, value);
    return;
  }

  window.localStorage.removeItem(key);
}

function createSnapshot() {
  const authToken = readStorage(AUTH_TOKEN_KEY);
  const tenantApiKey = readStorage(TENANT_API_KEY);
  const tenantId = readStorage(TENANT_ID_KEY) || "";

  return {
    authToken,
    hasTenantApiKey: Boolean(tenantApiKey),
    isAuthenticated: Boolean(authToken),
    tenantApiKey,
    tenantId,
  };
}

function snapshotsMatch(left, right) {
  return (
    left.authToken === right.authToken &&
    left.hasTenantApiKey === right.hasTenantApiKey &&
    left.isAuthenticated === right.isAuthenticated &&
    left.tenantApiKey === right.tenantApiKey &&
    left.tenantId === right.tenantId
  );
}

let authSnapshot = createSnapshot();

function emitAuthChange() {
  const nextSnapshot = createSnapshot();
  if (snapshotsMatch(authSnapshot, nextSnapshot)) {
    return;
  }

  authSnapshot = nextSnapshot;
  listeners.forEach((listener) => listener());
}

function updateStorage(key, value) {
  writeStorage(key, value);
  emitAuthChange();
}

export function subscribeAuthStore(listener) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getAuthSnapshot() {
  return authSnapshot;
}

export function getAuthServerSnapshot() {
  return emptySnapshot;
}

if (typeof window !== "undefined" && !window[STORAGE_LISTENER_KEY]) {
  window.addEventListener("storage", emitAuthChange);
  window[STORAGE_LISTENER_KEY] = true;
}

export function getAuthToken() {
  return getAuthSnapshot().authToken;
}

export function setAuthToken(token) {
  updateStorage(AUTH_TOKEN_KEY, token);
}

export function clearAuthToken() {
  updateStorage(AUTH_TOKEN_KEY, "");
}

export function isAuthenticated() {
  return getAuthSnapshot().isAuthenticated;
}

export function getTenantApiKey() {
  return getAuthSnapshot().tenantApiKey;
}

export function getTenantId() {
  return getAuthSnapshot().tenantId;
}

export function hasTenantApiKey() {
  return getAuthSnapshot().hasTenantApiKey;
}

export function setTenantContext({ tenantId = "", apiKey = "" }) {
  writeStorage(TENANT_ID_KEY, tenantId.trim());
  writeStorage(TENANT_API_KEY, apiKey.trim());
  emitAuthChange();
}

export function setTenantId(tenantId = "") {
  writeStorage(TENANT_ID_KEY, tenantId.trim());
  emitAuthChange();
}

export function setTenantApiKey(apiKey = "") {
  writeStorage(TENANT_API_KEY, apiKey.trim());
  emitAuthChange();
}

export function clearTenantContext() {
  writeStorage(TENANT_ID_KEY, "");
  writeStorage(TENANT_API_KEY, "");
  emitAuthChange();
}
