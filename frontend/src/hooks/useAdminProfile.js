import { useCallback, useEffect } from "react";
import { adminRequest } from "../api";
import { clearTenantContext, setTenantId } from "../auth";
import { useAsyncData } from "./useAsyncData";
import { useAuthSession } from "./useAuthSession";

const defaultProfile = { user: "Admin" };

export function useAdminProfile() {
  const { isAuthenticated } = useAuthSession();
  const loader = useCallback(({ signal }) => (
    adminRequest("/admin/auth/me", { signal })
  ), []);
  const profileState = useAsyncData(loader, {
    enabled: isAuthenticated,
    initialData: defaultProfile,
  });

  useEffect(() => {
    if (!isAuthenticated) {
      clearTenantContext();
      return;
    }

    if (profileState.loading) {
      return;
    }

    if (profileState.data?.tenant_id) {
      setTenantId(profileState.data.tenant_id);
      return;
    }

    clearTenantContext();
  }, [isAuthenticated, profileState.data?.tenant_id, profileState.loading]);

  return {
    ...profileState,
    profile: profileState.data || defaultProfile,
  };
}
