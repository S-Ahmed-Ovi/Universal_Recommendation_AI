"""
GET /algorithms

Returns metadata for every algorithm registered in ALGORITHM_REGISTRY.
Public endpoint — no auth required.
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.core.algorithms import ALGORITHM_REGISTRY
from backend.utils.logger import get_logger

log = get_logger(__name__)

router = APIRouter(tags=["Algorithms"])


@router.get("/algorithms")
def list_algorithms():
    """Return all available recommendation algorithms with their metadata."""
    return {
        "algorithms": [
            {
                "name": name,
                "feedback": meta.get("feedback"),
                "domain": meta.get("domain"),
                "description": meta.get("description"),
                "training_speed": meta.get("training_speed"),
                "scalability": meta.get("scalability"),
                "robustness": meta.get("robustness"),
                "interpretability": meta.get("interpretability"),
                "production_readiness": meta.get("production_readiness"),
            }
            for name, meta in ALGORITHM_REGISTRY.items()
        ]
    }