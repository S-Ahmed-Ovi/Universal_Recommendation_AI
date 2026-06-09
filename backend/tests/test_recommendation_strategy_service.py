from __future__ import annotations

from backend.core.models.model_loader import LoadedModel
from backend.core.models.model_registry import ModelRecord
from backend.core.recommendation.recommender_engine import RecommendationResult
from backend.core.recommendation.strategy_service import (
    RecommendationPayload,
    RecommendationStrategyError,
    RecommendationStrategyService,
    WeightedModelInput,
)
import backend.core.recommendation.strategy_service as strategy_module


class FakeRegistry:
    def __init__(self, records: list[ModelRecord], tenant_id: str = "tenant-a"):
        self.tenant_id = tenant_id
        self._records = {record.model_id: record for record in records}
        self._latest_by_algorithm: dict[str, ModelRecord] = {}
        for record in sorted(records, key=lambda item: item.created_at, reverse=True):
            self._latest_by_algorithm.setdefault(record.algorithm, record)

    def get(self, model_id: str):
        return self._records.get(model_id)

    def get_promoted(self, algorithm: str):
        for record in self._records.values():
            if record.algorithm == algorithm and record.promoted:
                return record
        return None

    def latest_for_algorithm(self, algorithm: str):
        return self._latest_by_algorithm.get(algorithm)

    def list_models(self):
        return sorted(self._records.values(), key=lambda item: item.created_at, reverse=True)


class FakeLoader:
    def __init__(self, registry: FakeRegistry):
        self.registry = registry

    def load_by_id(self, tenant_id: str, model_id: str) -> LoadedModel:
        record = self.registry.get(model_id)
        if record is None:
            raise RuntimeError(f"missing model: {model_id}")
        return LoadedModel(
            model={"model_id": model_id},
            record=record,
            tenant_id=tenant_id,
            loaded_at=1.0,
        )


class FakeRecommender:
    def __init__(self, outputs: dict[str, list[dict]], tenant_id: str = "tenant-a"):
        self.outputs = outputs
        self.tenant_id = tenant_id

    def recommend_with_loaded_model(self, user_id, loaded, top_n=10, algorithm_label=None, exclude_seen=True):
        return RecommendationResult(
            user_id=str(user_id),
            algorithm=algorithm_label or loaded.algorithm,
            model_id=loaded.model_id,
            recommendations=list(self.outputs.get(loaded.model_id, [])),
            top_n=top_n,
            generated_at=1.0,
            tenant_id=self.tenant_id,
        )


def make_record(
    model_id: str,
    algorithm: str,
    *,
    promoted: bool = False,
    created_at: float = 1.0,
) -> ModelRecord:
    return ModelRecord(
        model_id=model_id,
        algorithm=algorithm,
        version=1,
        metrics={"Composite Score": 0.99},
        params={},
        is_implicit=True,
        created_at=created_at,
        tenant_id="tenant-a",
        promoted=promoted,
        file_path=f"/tmp/{model_id}.pkl",
    )


def build_service(records: list[ModelRecord], outputs: dict[str, list[dict]]) -> RecommendationStrategyService:
    registry = FakeRegistry(records)
    loader = FakeLoader(registry)
    recommender = FakeRecommender(outputs)
    return RecommendationStrategyService(
        registry=registry,
        loader=loader,
        recommender=recommender,
        tenant_id="tenant-a",
    )


def test_ensemble_weighted_can_serve_saved_ranked_models(monkeypatch):
    monkeypatch.setattr(strategy_module, "ALLOW_INACTIVE_MODEL_SERVING", False)

    records = [
        make_record("model-pop", "Ecommerce-Popularity", created_at=10.0),
        make_record("model-p3", "P3Alpha", promoted=True, created_at=9.0),
    ]
    outputs = {
        "model-pop": [
            {"item_id": "item-1", "score": 0.9, "rank": 1},
            {"item_id": "item-2", "score": 0.4, "rank": 2},
        ],
        "model-p3": [
            {"item_id": "item-1", "score": 0.3, "rank": 1},
            {"item_id": "item-3", "score": 0.8, "rank": 2},
        ],
    }
    service = build_service(records, outputs)

    result = service.recommend(
        RecommendationPayload(
            user_id="user-1",
            top_n=5,
            strategy="ensemble_weighted",
            models=[
                WeightedModelInput(model_id="model-pop", algorithm="Ecommerce-Popularity", weight=0.6),
                WeightedModelInput(model_id="model-p3", algorithm="P3Alpha", weight=0.4),
            ],
        )
    )

    assert result["strategy"] == "ensemble_weighted"
    assert [row["model_id"] for row in result["models_used"]] == ["model-pop", "model-p3"]
    assert result["models_used"][0]["promoted"] is False
    assert result["models_used"][1]["promoted"] is True
    assert result["recommendations"]


def test_ensemble_weighted_algorithm_only_rows_fallback_to_latest_saved_models(monkeypatch):
    monkeypatch.setattr(strategy_module, "ALLOW_INACTIVE_MODEL_SERVING", False)

    records = [
        make_record("model-old", "EASE", created_at=5.0),
        make_record("model-new", "EASE", created_at=10.0),
        make_record("model-bpr", "BPR", created_at=9.0),
    ]
    outputs = {
        "model-new": [
            {"item_id": "item-1", "score": 0.8, "rank": 1},
            {"item_id": "item-4", "score": 0.2, "rank": 2},
        ],
        "model-bpr": [
            {"item_id": "item-2", "score": 0.7, "rank": 1},
            {"item_id": "item-1", "score": 0.5, "rank": 2},
        ],
    }
    service = build_service(records, outputs)

    result = service.recommend(
        RecommendationPayload(
            user_id="user-2",
            top_n=5,
            strategy="ensemble_weighted",
            models=[
                WeightedModelInput(algorithm="EASE", weight=0.5),
                WeightedModelInput(algorithm="BPR", weight=0.5),
            ],
        )
    )

    assert [row["model_id"] for row in result["models_used"]] == ["model-new", "model-bpr"]


def test_single_model_still_rejects_unpromoted_model_ids(monkeypatch):
    monkeypatch.setattr(strategy_module, "ALLOW_INACTIVE_MODEL_SERVING", False)

    service = build_service(
        [make_record("model-pop", "Ecommerce-Popularity", created_at=10.0)],
        {"model-pop": [{"item_id": "item-1", "score": 0.9, "rank": 1}]},
    )

    try:
        service.recommend(
            RecommendationPayload(
                user_id="user-3",
                top_n=5,
                strategy="single_model",
                model_id="model-pop",
                algorithm="Ecommerce-Popularity",
            )
        )
    except RecommendationStrategyError as exc:
        assert exc.code == "INACTIVE_MODEL_NOT_ALLOWED"
    else:
        raise AssertionError("single_model should still reject unpromoted model IDs")
