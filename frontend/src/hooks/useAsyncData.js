import { useCallback, useEffect, useRef, useState } from "react";

const defaultOptions = {
  enabled: true,
  initialData: null,
  resetOnDisable: true,
};

function isAbortError(error) {
  return error?.name === "AbortError";
}

export function useAsyncData(loader, options = {}) {
  const {
    enabled,
    initialData,
    resetOnDisable,
  } = {
    ...defaultOptions,
    ...options,
  };
  const [data, setData] = useState(initialData);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(Boolean(enabled));
  const initialDataRef = useRef(initialData);
  const mountedRef = useRef(false);
  const requestRef = useRef({ controller: null, id: 0 });

  useEffect(() => {
    initialDataRef.current = initialData;
  }, [initialData]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      requestRef.current.controller?.abort();
      requestRef.current.id += 1;
    };
  }, []);

  const reload = useCallback(async (reloadOptions = {}) => {
    if (!enabled) {
      requestRef.current.controller?.abort();
      if (mountedRef.current) {
        if (resetOnDisable) {
          setData(initialDataRef.current);
        }
        setError("");
        setLoading(false);
      }
      return initialDataRef.current;
    }

    requestRef.current.controller?.abort();
    const controller = new AbortController();
    const requestId = requestRef.current.id + 1;
    requestRef.current = { controller, id: requestId };

    if (mountedRef.current) {
      if (!reloadOptions.silent) {
        setLoading(true);
      }
      setError("");
    }

    try {
      const nextData = await loader({
        requestId,
        signal: controller.signal,
      });

      if (mountedRef.current && requestRef.current.id === requestId) {
        setData(nextData);
      }

      return nextData;
    } catch (loadError) {
      if (controller.signal.aborted || isAbortError(loadError)) {
        return initialDataRef.current;
      }

      if (mountedRef.current && requestRef.current.id === requestId) {
        setError(loadError?.message || "Unable to load data");
      }

      throw loadError;
    } finally {
      if (mountedRef.current && requestRef.current.id === requestId) {
        setLoading(false);
      }
    }
  }, [enabled, loader, resetOnDisable]);

  useEffect(() => {
    if (!enabled) {
      requestRef.current.controller?.abort();
      if (resetOnDisable) {
        setData(initialDataRef.current);
      }
      setError("");
      setLoading(false);
      return undefined;
    }

    reload().catch(() => {});
    return () => {
      requestRef.current.controller?.abort();
    };
  }, [enabled, reload, resetOnDisable]);

  return {
    data,
    error,
    loading,
    reload,
    setData,
    setError,
  };
}
