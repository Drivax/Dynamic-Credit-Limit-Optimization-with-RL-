"""Compatibility entry point for the packaged PD experiment."""
from credit_rl.experiments.pd_training import evaluate, main, run

__all__ = ["evaluate", "main", "run"]

if __name__ == "__main__":
    main()
