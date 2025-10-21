from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from market_nn_plus_ultra.service import ServiceSettings, create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the Market NN Plus Ultra inference API")
    parser.add_argument("--config", type=Path, required=True, help="Path to the experiment YAML config")
    parser.add_argument("--checkpoint", type=Path, help="Optional Lightning checkpoint to restore")
    parser.add_argument("--device", type=str, default="cpu", help="Device identifier (cpu, cuda:0, etc.)")
    parser.add_argument(
        "--return-column",
        type=str,
        default="realised_return",
        help="Return column used when computing evaluation metrics",
    )
    parser.add_argument(
        "--max-prediction-rows",
        type=int,
        default=1024,
        help="Maximum number of prediction rows the API will return per request",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface for the server")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development only)")
    parser.add_argument("--no-eval", action="store_true", help="Disable evaluation metrics by default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = ServiceSettings(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        evaluate_by_default=not args.no_eval,
        return_column=args.return_column,
        max_prediction_rows=args.max_prediction_rows if args.max_prediction_rows > 0 else None,
    )
    app = create_app(settings)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
