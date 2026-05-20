"""
Thin WandB wrapper so the Trainer never imports wandb directly.

If wandb.enabled is False in config (or the WANDB_DISABLED env var is set),
all calls become no-ops — useful for CI, offline Kaggle runs, or local debug.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class WandBLogger:
    """Wraps wandb init / log / finish with a clean enabled/disabled toggle.

    Args:
        enabled:  Whether to actually log to WandB.
        project:  WandB project name.
        entity:   WandB username or team (None = personal default).
        run_name: Display name for this run (e.g. 'densenet121_full').
        tags:     List of tag strings.
        config:   Flat dict of hyperparameters to record at run start.
    """

    def __init__(
        self,
        enabled: bool,
        project: str,
        entity: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[list] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.enabled = enabled
        self._run = None

        if not enabled:
            return

        import wandb  # imported lazily so the rest of the codebase works without it

        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            tags=tags or [],
            config=config or {},
            resume="allow",  # safe to call on a resumed training run
        )

    # ── Logging ──────────────────────────────────────────────────────────────

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log a dict of scalars. step is the global training step or epoch."""
        if not self.enabled or self._run is None:
            return
        self._run.log(metrics, step=step)

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """Convenience wrapper: prefix all keys with 'epoch/' and log."""
        prefixed = {f"epoch/{k}": v for k, v in metrics.items()}
        prefixed["epoch"] = epoch
        self.log(prefixed, step=epoch)

    def log_batch(self, step: int, metrics: Dict[str, Any]) -> None:
        """Log per-batch metrics (loss, lr) keyed under 'batch/'."""
        prefixed = {f"batch/{k}": v for k, v in metrics.items()}
        self.log(prefixed, step=step)

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def watch(self, model, log_freq: int = 100) -> None:
        """Attach gradient/parameter histograms to the run."""
        if not self.enabled or self._run is None:
            return
        import wandb
        wandb.watch(model, log="all", log_freq=log_freq)

    def finish(self) -> None:
        """Mark the run as finished (call at end of training)."""
        if not self.enabled or self._run is None:
            return
        self._run.finish()

    @property
    def run_url(self) -> Optional[str]:
        if self._run is None:
            return None
        return self._run.url
