"""
api/routes package

Import all routers here so app_factory.py can register them with a
single include loop:

    from backend.api.routes import routers
    for r in routers:
        app.include_router(r, prefix="/api/v3")
"""
from .algorithms import router as algorithms_router
from .context import router as context_router
from .training import router as training_router
from .recommend import router as recommend_router
from .batch import router as batch_router
from .similar import router as similar_router
from .explain import router as explain_router
from .models import router as models_router
from .jobs import router as jobs_router
from .scheduler import router as scheduler_router

routers = [
    algorithms_router,
    context_router,
    training_router,
    recommend_router,
    batch_router,
    similar_router,
    explain_router,
    models_router,
    jobs_router,
    scheduler_router,
]

__all__ = [
    "algorithms_router",
    "context_router",
    "training_router",
    "recommend_router",
    "batch_router",
    "similar_router",
    "explain_router",
    "models_router",
    "jobs_router",
    "scheduler_router",
    "routers",
]