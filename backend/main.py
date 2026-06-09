"""
Application entrypoint for CLI and FastAPI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from colorama import Fore, Style

# Allow `uvicorn main:app` from inside `backend/` by ensuring the repo root is on sys.path.
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from backend.app_factory import create_app
from backend.config.settings import (
    API_HOST,
    API_PORT,
    DEFAULT_ALGORITHM_MODE,
    OPTUNA_TRIALS_AUTO,
    TOP_K,
    TOP_N_MODELS,
)
from backend.utils.logger import get_logger

log = get_logger(__name__)
app = create_app()


def main():
    """Parse CLI flags and either launch the API or run local training."""
    parser = argparse.ArgumentParser(description='Proactive AI - CLI TRAINING')
    parser.add_argument('--file', type=str, help='Path to data file')
    parser.add_argument('--format', type=str, default='auto', help='file format: csv|excel|json|parquet|auto')
    parser.add_argument('--topk', type=int, default=TOP_K, help='top-k metric cutoff (allowed: 5 or 10)')
    parser.add_argument('--top-models', type=int, default=TOP_N_MODELS, help='number of top ranked models to tune/save (allowed: 5 or 10)')
    parser.add_argument('--trials', type=int, default=OPTUNA_TRIALS_AUTO, help='Optuna trials (-1=adaptive)')
    parser.add_argument('--mode', type=str, default=DEFAULT_ALGORITHM_MODE, help='algorithm mode: explicit|implicit|auto')
    parser.add_argument('--interactive', action='store_true', help='interactive column mapping')
    parser.add_argument('--force-all', action='store_true', help='force wider benchmark coverage')
    parser.add_argument('--no-tune', action='store_true', help='skip Optuna')
    parser.add_argument('--serve', action='store_true', help='start FastAPI server')
    args = parser.parse_args()

    if args.serve or not args.file:
        _start_server()
        return
    _run_cli(args)


def _run_cli(args):
    from backend.core.pipeline.training_pipeline import TrainingConfig, TrainingPipeline

    print(f"\n{Fore.CYAN}{'=' * 70}")
    print('  PROACTIVE AI - CLI TRAINING')
    print(f"{'=' * 70}{Style.RESET_ALL}")

    config = TrainingConfig(
        top_k=args.topk,
        n_tuning_trials=0 if args.no_tune else args.trials,
        top_model_count=args.top_models,
        algorithm_mode=args.mode,
        force_all_algos=args.force_all,
        interactive=args.interactive,
        save_model=True,
        auto_promote=True,
    )
    result = TrainingPipeline(config).run_from_file(args.file, file_format=args.format)
    if result.best_algorithm:
        print(f"\n{Fore.GREEN}Best model: {result.best_algorithm}{Style.RESET_ALL}")
        print(f'Model ID: {result.best_model_id}')
        print(f'Params: {result.best_params}')


def _start_server():
    import uvicorn

    log.info('Starting API on %s:%d', API_HOST, API_PORT)
    print(f"\n{Fore.CYAN}Starting API: http://{API_HOST}:{API_PORT}/{Style.RESET_ALL}\n")
    uvicorn.run('backend.main:app', host=API_HOST, port=API_PORT, reload=True)


if __name__ == '__main__':
    main()
    # _start_server()
