"""
sdk/webhooks/push.py — Event-driven webhook delivery

New in v3.

Delivers signed JSON payloads to tenant-registered webhook URLs whenever
key lifecycle events occur inside the Proactive AI platform.

── Events ────────────────────────────────────────────────────────────────────

  Event name               Fired when
  ──────────────────────── ───────────────────────────────────────────────────
  job.completed            An async training job finishes successfully
  job.failed               An async training job terminates with an error
  model.promoted           A model is manually promoted via the API
  recommend.served         A recommendation is returned (sampled, opt-in only)
  recommend.batch.served   A batch recommendation is returned (sampled)

── Delivery guarantees ───────────────────────────────────────────────────────

  - At-least-once delivery with configurable retry (default: 3 attempts).
  - Exponential back-off between retries (1s → 2s → 4s …).
  - Delivery is async: the caller submits a fire-and-forget task to a
    background ThreadPoolExecutor; the API route is never blocked.
  - Each delivery attempt is logged with the HTTP status received.
  - Failures after all retries are logged as errors and emitted to the
    optional `on_failure` callback.

── Signature verification ────────────────────────────────────────────────────

  Every request carries an X-ProactiveAI-Signature header:

    X-ProactiveAI-Signature: sha256=<hex_digest>

  The signature is HMAC-SHA256 over the raw JSON body bytes using the
  tenant's webhook_secret.  Recipients should verify:

    import hashlib, hmac
    expected = hmac.new(secret.encode(), body_bytes, hashlib.sha256).hexdigest()
    received = request.headers["X-ProactiveAI-Signature"].removeprefix("sha256=")
    assert hmac.compare_digest(expected, received)

── Usage ─────────────────────────────────────────────────────────────────────

  # Fire from a route handler (non-blocking)
  from backend.sdk.webhooks.push import WebhookPusher, WebhookEvent

  pusher = WebhookPusher()                      # singleton — reuse across requests

  pusher.send(
      WebhookEvent(
          event     = "job.completed",
          tenant_id = "acme-corp",
          payload   = {"job_id": "abc123", "result": {...}},
      ),
      url    = tenant_config["webhook_url"],
      secret = tenant_config["webhook_secret"],
  )

  # Blocking delivery (e.g. tests)
  receipt = pusher.send_sync(event, url, secret)
  print(receipt.success, receipt.attempts)

  # Register a global failure callback
  pusher.on_failure = lambda event, receipt: alert_ops(event, receipt)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

DEFAULT_TIMEOUT_S    = 10
DEFAULT_MAX_RETRIES  = 3
DEFAULT_BACKOFF_BASE = 1.0   # seconds; doubles each retry

WEBHOOK_VERSION = "v1"
SIGNATURE_HEADER = "X-ProactiveAI-Signature"
EVENT_HEADER     = "X-ProactiveAI-Event"
VERSION_HEADER   = "X-ProactiveAI-Version"
DELIVERY_HEADER  = "X-ProactiveAI-Delivery"

VALID_EVENTS: frozenset[str] = frozenset({
    "job.completed",
    "job.failed",
    "model.promoted",
    "recommend.served",
    "recommend.batch.served",
})


# ── Data models ────────────────────────────────────────────────────────────────

@dataclass
class WebhookEvent:
    """
    A single platform event to be pushed to a tenant webhook.

    Parameters
    ----------
    event       : One of the VALID_EVENTS strings.
    tenant_id   : Owning tenant (used for logging).
    payload     : Arbitrary JSON-serialisable dict of event data.
    event_id    : Unique delivery ID (auto-generated if omitted).
    timestamp   : Unix epoch float (auto-set if omitted).
    """

    event: str
    tenant_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: _new_id())
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.event not in VALID_EVENTS:
            raise ValueError(
                f"Unknown event '{self.event}'. Valid events: {sorted(VALID_EVENTS)}"
            )

    def to_body(self) -> bytes:
        """Serialise to JSON bytes for delivery."""
        envelope = {
            "id":        self.event_id,
            "event":     self.event,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp,
            "version":   WEBHOOK_VERSION,
            "data":      self.payload,
        }
        return json.dumps(envelope, default=str, separators=(",", ":")).encode("utf-8")


@dataclass
class DeliveryReceipt:
    """Result of one webhook delivery attempt sequence."""

    event_id: str
    event: str
    tenant_id: str
    url: str
    success: bool
    attempts: int
    status_codes: list[int] = field(default_factory=list)
    final_error: Optional[str] = None
    elapsed_s: float = 0.0


# ── Signature helper ───────────────────────────────────────────────────────────

def _sign(body: bytes, secret: str) -> str:
    """Return 'sha256=<hex>' HMAC signature for the given body."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_signature(body: bytes, secret: str, header_value: str) -> bool:
    """
    Verify an inbound signature header.

    Constant-time comparison prevents timing attacks.

    Parameters
    ----------
    body         : raw request body bytes
    secret       : shared secret string
    header_value : the value of X-ProactiveAI-Signature from the request

    Returns
    -------
    bool  True if the signature is valid.
    """
    expected = _sign(body, secret)
    return hmac.compare_digest(expected, header_value)


# ── ID generator ───────────────────────────────────────────────────────────────

def _new_id() -> str:
    import uuid
    return str(uuid.uuid4())


# ── Core pusher ────────────────────────────────────────────────────────────────

class WebhookPusher:
    """
    Delivers WebhookEvent objects to tenant-registered HTTPS endpoints.

    Thread-safe.  Uses a background ThreadPoolExecutor for async delivery
    so API route handlers are never blocked.

    Parameters
    ----------
    max_retries    : Number of delivery attempts before giving up.
    timeout_s      : Per-attempt HTTP timeout in seconds.
    backoff_base   : Base seconds for exponential back-off between retries.
    max_workers    : Thread pool size for background delivery.
    on_failure     : Optional callback invoked after all retries are exhausted.
                     Signature: (event: WebhookEvent, receipt: DeliveryReceipt) -> None
    """

    def __init__(
        self,
        max_retries: int = DEFAULT_MAX_RETRIES,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        backoff_base: float = DEFAULT_BACKOFF_BASE,
        max_workers: int = 4,
        on_failure: Optional[Callable[[WebhookEvent, DeliveryReceipt], None]] = None,
    ) -> None:
        """Configure background delivery retries, timeouts, and failure callbacks."""
        self.max_retries   = max_retries
        self.timeout_s     = timeout_s
        self.backoff_base  = backoff_base
        self.on_failure    = on_failure
        self._pool         = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="webhook")

    # ── Public API ─────────────────────────────────────────────────────────────

    def send(
        self,
        event: WebhookEvent,
        url: str,
        secret: str,
    ) -> None:
        """
        Submit a non-blocking (fire-and-forget) delivery to the thread pool.

        Returns immediately.  Errors are logged; the optional on_failure
        callback is invoked after all retries are exhausted.
        """
        self._pool.submit(self._deliver, event, url, secret)

    def send_sync(
        self,
        event: WebhookEvent,
        url: str,
        secret: str,
    ) -> DeliveryReceipt:
        """
        Blocking delivery — useful for tests and CLI tooling.

        Returns a DeliveryReceipt describing the outcome.
        """
        return self._deliver(event, url, secret)

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the background thread pool gracefully."""
        self._pool.shutdown(wait=wait)

    # ── Delivery logic ────────────────────────────────────────────────────────

    def _deliver(
        self,
        event: WebhookEvent,
        url: str,
        secret: str,
    ) -> DeliveryReceipt:
        body         = event.to_body()
        signature    = _sign(body, secret)
        status_codes: list[int] = []
        final_error: Optional[str] = None
        t_start = time.monotonic()

        for attempt in range(1, self.max_retries + 1):
            try:
                status = self._http_post(url, body, signature, event)
                status_codes.append(status)

                if 200 <= status < 300:
                    elapsed = round(time.monotonic() - t_start, 3)
                    log.info(
                        "Webhook delivered | event=%s tenant=%s id=%s url=%s status=%d attempts=%d elapsed=%.3fs",
                        event.event, event.tenant_id, event.event_id, url,
                        status, attempt, elapsed,
                    )
                    return DeliveryReceipt(
                        event_id=event.event_id,
                        event=event.event,
                        tenant_id=event.tenant_id,
                        url=url,
                        success=True,
                        attempts=attempt,
                        status_codes=status_codes,
                        elapsed_s=elapsed,
                    )

                # Treat 4xx (except 429) as permanent failures — no retry
                if 400 <= status < 500 and status != 429:
                    final_error = f"Permanent failure: HTTP {status}"
                    log.warning(
                        "Webhook permanent failure | event=%s tenant=%s id=%s url=%s status=%d",
                        event.event, event.tenant_id, event.event_id, url, status,
                    )
                    break

                final_error = f"HTTP {status}"
                log.warning(
                    "Webhook attempt %d/%d failed | event=%s tenant=%s id=%s url=%s status=%d",
                    attempt, self.max_retries,
                    event.event, event.tenant_id, event.event_id, url, status,
                )

            except Exception as exc:
                final_error = str(exc)
                status_codes.append(0)
                log.warning(
                    "Webhook attempt %d/%d error | event=%s tenant=%s id=%s url=%s error=%s",
                    attempt, self.max_retries,
                    event.event, event.tenant_id, event.event_id, url, exc,
                )

            # Back-off before next attempt
            if attempt < self.max_retries:
                sleep = self.backoff_base * (2 ** (attempt - 1))
                time.sleep(sleep)

        elapsed = round(time.monotonic() - t_start, 3)
        receipt = DeliveryReceipt(
            event_id=event.event_id,
            event=event.event,
            tenant_id=event.tenant_id,
            url=url,
            success=False,
            attempts=len(status_codes),
            status_codes=status_codes,
            final_error=final_error,
            elapsed_s=elapsed,
        )

        log.error(
            "Webhook delivery failed after %d attempts | event=%s tenant=%s id=%s url=%s last_error=%s",
            receipt.attempts, event.event, event.tenant_id, event.event_id, url, final_error,
        )

        if callable(self.on_failure):
            try:
                self.on_failure(event, receipt)
            except Exception as cb_exc:
                log.error("on_failure callback raised: %s", cb_exc)

        return receipt

    def _http_post(
        self,
        url: str,
        body: bytes,
        signature: str,
        event: WebhookEvent,
    ) -> int:
        """
        Execute a single HTTP POST.  Returns the HTTP status code.
        Uses stdlib urllib so there are no extra dependencies.
        """
        headers = {
            "Content-Type":  "application/json; charset=utf-8",
            "User-Agent":    f"ProactiveAI-Webhook/{WEBHOOK_VERSION}",
            SIGNATURE_HEADER: signature,
            EVENT_HEADER:    event.event,
            VERSION_HEADER:  WEBHOOK_VERSION,
            DELIVERY_HEADER: event.event_id,
        }

        req = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.status
        except urllib.error.HTTPError as exc:
            return exc.code
        except urllib.error.URLError as exc:
            raise ConnectionError(f"URL error reaching {url}: {exc.reason}") from exc


# ── Module-level singleton ─────────────────────────────────────────────────────

#: Shared pusher instance — import and use directly in route handlers.
pusher = WebhookPusher()


# ── Convenience helpers ────────────────────────────────────────────────────────

def emit_job_completed(
    tenant_id: str,
    job_id: str,
    result: dict[str, Any],
    webhook_url: str,
    webhook_secret: str,
) -> None:
    """Fire a job.completed event (non-blocking)."""
    pusher.send(
        WebhookEvent(
            event="job.completed",
            tenant_id=tenant_id,
            payload={"job_id": job_id, "result": result},
        ),
        url=webhook_url,
        secret=webhook_secret,
    )


def emit_job_failed(
    tenant_id: str,
    job_id: str,
    error: str,
    webhook_url: str,
    webhook_secret: str,
) -> None:
    """Fire a job.failed event (non-blocking)."""
    pusher.send(
        WebhookEvent(
            event="job.failed",
            tenant_id=tenant_id,
            payload={"job_id": job_id, "error": error},
        ),
        url=webhook_url,
        secret=webhook_secret,
    )


def emit_model_promoted(
    tenant_id: str,
    model_id: str,
    algorithm: str,
    webhook_url: str,
    webhook_secret: str,
) -> None:
    """Fire a model.promoted event (non-blocking)."""
    pusher.send(
        WebhookEvent(
            event="model.promoted",
            tenant_id=tenant_id,
            payload={"model_id": model_id, "algorithm": algorithm},
        ),
        url=webhook_url,
        secret=webhook_secret,
    )
