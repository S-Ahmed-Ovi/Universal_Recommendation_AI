"""
smart_db_csv_builder/api — FastAPI route layer for the SmartDB CSV Builder.

Exposes four routers:
    connections  — register / list database connections
    schema       — discover table & column metadata
    build        — trigger an LLM-planned or manual CSV build job
    jobs         — poll async job status

All routers are mounted by app_factory.py under /smart-db-csv/api/
and require the training:write scope (enforced at the router level in
app_factory, and additionally on write endpoints here).
"""
