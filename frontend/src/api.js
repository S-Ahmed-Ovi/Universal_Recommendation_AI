import {
  API_BASE_URL,
  clearAuthToken,
  getAuthToken,
  getTenantApiKey,
} from "./auth";

function isFormData(value) {
  return typeof FormData !== "undefined" && value instanceof FormData;
}

export class ApiError extends Error {
  constructor(message, context = {}) {
    super(message);
    this.name = "ApiError";
    this.path = context.path || "";
    this.payload = context.payload;
    this.status = context.status || 0;
  }
}

async function readResponse(response) {
  const text = await response.text();
  if (!text) {
    return null;
  }

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function errorMessage(response, payload) {
  if (typeof payload === "string" && payload.trim()) {
    return payload;
  }

  if (payload && typeof payload === "object" && payload.detail) {
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (typeof payload.detail.message === "string") {
      return payload.detail.message;
    }
    return JSON.stringify(payload.detail);
  }

  return `${response.status} ${response.statusText}`;
}

async function request(path, options = {}) {
  const {
    auth = "public",
    body,
    headers = {},
    method = "GET",
    signal,
  } = options;

  const finalHeaders = new Headers(headers);

  if (auth === "admin") {
    const token = getAuthToken();
    if (!token) {
      throw new Error("Admin login is required");
    }
    finalHeaders.set("Authorization", `Bearer ${token}`);
  }

  if (auth === "tenant") {
    const apiKey = getTenantApiKey();
    if (!apiKey) {
      throw new Error("Tenant API key is required");
    }
    finalHeaders.set("X-API-Key", apiKey);
  }

  let finalBody = body;
  if (body && !isFormData(body) && typeof body === "object") {
    finalHeaders.set("Content-Type", "application/json");
    finalBody = JSON.stringify(body);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    body: finalBody,
    headers: finalHeaders,
    method,
    signal,
  });

  const payload = await readResponse(response);
  if (!response.ok) {
    if (auth === "admin" && response.status === 401) {
      clearAuthToken();
    }
    throw new ApiError(errorMessage(response, payload), {
      path,
      payload,
      status: response.status,
    });
  }

  return payload;
}

export function publicRequest(path, options) {
  return request(path, { ...options, auth: "public" });
}

export function adminRequest(path, options) {
  return request(path, { ...options, auth: "admin" });
}

export function tenantRequest(path, options) {
  return request(path, { ...options, auth: "tenant" });
}

export function tenantContextRequest() {
  return request("/tenant/context", { auth: "tenant" });
}

export function verifyTenantApiKey(apiKey) {
  return request("/tenant/context", {
    auth: "public",
    headers: {
      "X-API-Key": apiKey,
    },
  });
}

export function formatTimestamp(value) {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }

  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return String(value);
  }

  const epochMs = numeric < 1e12 ? numeric * 1000 : numeric;
  return new Date(epochMs).toLocaleString();
}

export function maskSecret(value) {
  if (!value) {
    return "Not configured";
  }

  if (value.length <= 10) {
    return value;
  }

  return `${value.slice(0, 6)}...${value.slice(-4)}`;
}
