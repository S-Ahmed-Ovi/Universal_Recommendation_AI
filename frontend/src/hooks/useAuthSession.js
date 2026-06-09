import { useSyncExternalStore } from "react";
import {
  getAuthServerSnapshot,
  getAuthSnapshot,
  subscribeAuthStore,
} from "../auth";

export function useAuthSession() {
  return useSyncExternalStore(
    subscribeAuthStore,
    getAuthSnapshot,
    getAuthServerSnapshot,
  );
}

export function useTenantSession() {
  const session = useAuthSession();

  return {
    hasTenantApiKey: session.hasTenantApiKey,
    tenantApiKey: session.tenantApiKey,
    tenantId: session.tenantId,
  };
}
