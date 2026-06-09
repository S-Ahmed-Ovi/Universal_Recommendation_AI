"""
smart_db_csv_builder/connectors/nosql.py — NoSQL connector

Moved from ingestion/nosql_connector.py (v2).

v3 changes (minimal — "move as-is" per migration plan)
-------------------------------------------------------
- tenant_id parameter added to constructor and all log lines.
- Module import path updated to smart_db_csv_builder.connectors.nosql.
- All backend logic (MongoDB, Cassandra, DynamoDB) is unchanged.

Supports: MongoDB, Cassandra, DynamoDB
Each backend driver is lazily imported so users only need to install
the driver they actually use.

Usage
-----
    from backend.config.database import NoSQLConfig

    cfg = NoSQLConfig(engine="mongodb", host="localhost", port=27017,
                      database="shop")
    with NoSQLConnector(cfg, tenant_id="acme-corp") as conn:
        df = conn.fetch_collection(
            "interactions",
            query={"rating": {"$gte": 1}},
            limit=50_000,
        )
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backend.config.database import NoSQLConfig
from backend.utils.logger import get_logger

log = get_logger(__name__)


class NoSQLConnector:
    """
    Unified NoSQL connector. Backend is selected from NoSQLConfig.engine.

    Supported engines: mongodb, cassandra, dynamodb
    """

    def __init__(self, config: NoSQLConfig, tenant_id: str = "") -> None:
        """Store tenant-aware NoSQL connection settings until a backend is selected."""
        self.config    = config
        self.tenant_id = tenant_id
        self._client   = None

    def __enter__(self) -> "NoSQLConnector":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    # ── Connection ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Open the configured NoSQL backend using the engine-specific adapter."""
        engine = self.config.engine.lower()
        if engine == "mongodb":
            self._connect_mongo()
        elif engine == "cassandra":
            self._connect_cassandra()
        elif engine == "dynamodb":
            self._connect_dynamodb()
        else:
            raise ValueError(
                f"Unsupported NoSQL engine: '{engine}'. "
                "Choose: mongodb, cassandra, dynamodb"
            )

    def disconnect(self) -> None:
        """Close the active NoSQL client if the backend exposes a close method."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            log.info(
                "NoSQL disconnected | tenant=%s engine=%s",
                self.tenant_id or "–", self.config.engine,
            )

    # ── MongoDB ────────────────────────────────────────────────────────────

    def _connect_mongo(self) -> None:
        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("Install pymongo: pip install pymongo")

        uri = self._build_mongo_uri()
        self._client = MongoClient(
            uri,
            serverSelectionTimeoutMS=self.config.connect_timeout * 1000,
        )
        self._client.server_info()  # raises if unreachable
        log.info(
            "MongoDB connected | tenant=%s host=%s db=%s",
            self.tenant_id or "–", self.config.host, self.config.database,
        )

    def _build_mongo_uri(self) -> str:
        cfg = self.config
        if cfg.username and cfg.password:
            return (
                f"mongodb://{cfg.username}:{cfg.password}"
                f"@{cfg.host}:{cfg.port}/{cfg.database}"
            )
        return f"mongodb://{cfg.host}:{cfg.port}/{cfg.database}"

    def fetch_collection(
        self,
        collection: str,
        query: Optional[dict] = None,
        projection: Optional[dict] = None,
        limit: int = 0,
    ) -> pd.DataFrame:
        """
        Fetch documents from a MongoDB collection into a DataFrame.

        Parameters
        ----------
        collection : collection name
        query      : MongoDB filter dict (default: all documents)
        projection : fields to include/exclude
        limit      : max documents to return (0 = unlimited)
        """
        db  = self._client[self.config.database]
        col = db[collection]
        cur = col.find(query or {}, projection or {}, limit=limit)
        docs = list(cur)
        df   = pd.DataFrame(docs)
        if "_id" in df.columns:
            df["_id"] = df["_id"].astype(str)
        log.info(
            "MongoDB fetch | tenant=%s collection=%s docs=%d",
            self.tenant_id or "–", collection, len(df),
        )
        return df

    # ── Cassandra ──────────────────────────────────────────────────────────

    def _connect_cassandra(self) -> None:
        try:
            from cassandra.cluster import Cluster
            from cassandra.auth import PlainTextAuthProvider
        except ImportError:
            raise ImportError("Install cassandra-driver: pip install cassandra-driver")

        auth = None
        if self.config.username and self.config.password:
            auth = PlainTextAuthProvider(self.config.username, self.config.password)

        cluster      = Cluster([self.config.host], port=self.config.port, auth_provider=auth)
        self._client = cluster.connect(self.config.database)
        log.info(
            "Cassandra connected | tenant=%s host=%s keyspace=%s",
            self.tenant_id or "–", self.config.host, self.config.database,
        )

    def fetch_cassandra_table(
        self,
        table: str,
        cql: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Execute CQL and return a DataFrame.

        Parameters
        ----------
        table : table (or view) name — used when cql is None
        cql   : optional explicit CQL string; defaults to SELECT * FROM table
        """
        sql  = cql or f"SELECT * FROM {table}"
        rows = self._client.execute(sql)
        df   = pd.DataFrame(rows)
        log.info(
            "Cassandra fetch | tenant=%s table=%s rows=%d",
            self.tenant_id or "–", table, len(df),
        )
        return df

    # ── DynamoDB ───────────────────────────────────────────────────────────

    def _connect_dynamodb(self) -> None:
        try:
            import boto3
        except ImportError:
            raise ImportError("Install boto3: pip install boto3")

        region       = self.config.region or "us-east-1"
        self._client = boto3.resource("dynamodb", region_name=region)
        log.info(
            "DynamoDB connected | tenant=%s region=%s",
            self.tenant_id or "–", region,
        )


    # ── Schema introspection ───────────────────────────────────────────────

    def get_collection_names(self) -> list[str]:
        """
        Return all collection / table names in the connected database.

        - MongoDB   → db.list_collection_names()
        - Cassandra → SELECT table_name FROM system_schema.tables
        - DynamoDB  → paginated ListTables API call
        """
        self._require_connected()
        engine = self.config.engine.lower()

        if engine == "mongodb":
            db = self._client[self.config.database]
            names = db.list_collection_names()
            log.debug(
                "MongoDB collections | tenant=%s count=%d",
                self.tenant_id or "-", len(names),
            )
            return sorted(names)

        if engine == "cassandra":
            rows = self._client.execute(
                "SELECT table_name FROM system_schema.tables WHERE keyspace_name = %s",
                (self.config.database,),
            )
            names = [r.table_name for r in rows]
            log.debug(
                "Cassandra tables | tenant=%s keyspace=%s count=%d",
                self.tenant_id or "-", self.config.database, len(names),
            )
            return sorted(names)

        if engine == "dynamodb":
            names: list[str] = []
            kwargs: dict = {}
            while True:
                resp = self._client.meta.client.list_tables(**kwargs)
                names.extend(resp.get("TableNames", []))
                last = resp.get("LastEvaluatedTableName")
                if not last:
                    break
                kwargs["ExclusiveStartTableName"] = last
            log.debug(
                "DynamoDB tables | tenant=%s count=%d",
                self.tenant_id or "-", len(names),
            )
            return sorted(names)

        raise ValueError(f"Unsupported engine for introspection: {engine!r}")

    def get_columns(self, collection: str) -> list[dict]:
        """
        Infer column/field metadata by sampling documents from *collection*.

        NoSQL databases have no fixed schema, so this method samples up to
        100 documents and unions all observed field names.

        Returns a list of dicts: {name, type, nullable, primary_key}

        - MongoDB   → sample via find(limit=100)
        - Cassandra → query system_schema.columns for the table
        - DynamoDB  → sample via scan(Limit=100)
        """
        self._require_connected()
        engine = self.config.engine.lower()

        if engine == "mongodb":
            db   = self._client[self.config.database]
            col  = db[collection]
            docs = list(col.find({}, limit=100))
            if not docs:
                return []
            # Union all field names across sampled documents
            all_keys: dict[str, set] = {}
            for doc in docs:
                for k, v in doc.items():
                    all_keys.setdefault(k, set()).add(type(v).__name__)
            return [
                {
                    "name":        k,
                    "type":        "/".join(sorted(types)),
                    "nullable":    True,
                    "primary_key": k == "_id",
                }
                for k, types in all_keys.items()
            ]

        if engine == "cassandra":
            rows = self._client.execute(
                "SELECT column_name, type, kind FROM system_schema.columns "                "WHERE keyspace_name = %s AND table_name = %s",
                (self.config.database, collection),
            )
            return [
                {
                    "name":        r.column_name,
                    "type":        r.type,
                    "nullable":    r.kind not in ("partition_key", "clustering"),
                    "primary_key": r.kind in ("partition_key", "clustering"),
                }
                for r in rows
            ]

        if engine == "dynamodb":
            table = self._client.Table(collection)
            resp  = table.scan(Limit=100)
            items = resp.get("Items", [])
            if not items:
                # Fall back to key schema from table description
                desc = table.meta.client.describe_table(TableName=collection)
                key_schema = desc["Table"]["KeySchema"]
                attr_defs  = {
                    a["AttributeName"]: a["AttributeType"]
                    for a in desc["Table"]["AttributeDefinitions"]
                }
                return [
                    {
                        "name":        k["AttributeName"],
                        "type":        attr_defs.get(k["AttributeName"], "S"),
                        "nullable":    False,
                        "primary_key": True,
                    }
                    for k in key_schema
                ]
            all_keys: dict[str, set] = {}
            for item in items:
                for k, v in item.items():
                    all_keys.setdefault(k, set()).add(type(v).__name__)
            # Identify key attributes from table description
            desc       = table.meta.client.describe_table(TableName=collection)
            pk_names   = {k["AttributeName"] for k in desc["Table"]["KeySchema"]}
            return [
                {
                    "name":        k,
                    "type":        "/".join(sorted(types)),
                    "nullable":    k not in pk_names,
                    "primary_key": k in pk_names,
                }
                for k, types in all_keys.items()
            ]

        raise ValueError(f"Unsupported engine for column introspection: {engine!r}")

    def get_sample(self, collection: str, n: int = 5) -> pd.DataFrame:
        """
        Return the first *n* documents/rows of *collection* as a DataFrame.
        Used by the LLM planner for schema-aware column detection.

        - MongoDB   → fetch_collection(limit=n)
        - Cassandra → fetch_cassandra_table with LIMIT clause
        - DynamoDB  → fetch_dynamodb_table with Limit kwarg
        """
        self._require_connected()
        engine = self.config.engine.lower()

        if engine == "mongodb":
            return self.fetch_collection(collection, limit=n)

        if engine == "cassandra":
            return self.fetch_cassandra_table(
                collection,
                cql=f"SELECT * FROM {collection} LIMIT {int(n)}",
            )

        if engine == "dynamodb":
            return self.fetch_dynamodb_table(
                collection,
                scan_kwargs={"Limit": int(n)},
            )

        raise ValueError(f"Unsupported engine for get_sample: {engine!r}")

    # ── Internal ───────────────────────────────────────────────────────────

    def _require_connected(self) -> None:
        if self._client is None:
            raise RuntimeError(
                "Not connected. Call connect() or use the context manager."
            )
    def fetch_dynamodb_table(
        self,
        table_name: str,
        scan_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        Full-scan a DynamoDB table (pagination handled automatically).

        Parameters
        ----------
        table_name  : DynamoDB table name
        scan_kwargs : extra kwargs forwarded to Table.scan()
                      (e.g. FilterExpression, ProjectionExpression)
        """
        table  = self._client.Table(table_name)
        kwargs = scan_kwargs or {}
        items: list[dict] = []

        while True:
            resp  = table.scan(**kwargs)
            items.extend(resp.get("Items", []))
            last  = resp.get("LastEvaluatedKey")
            if not last:
                break
            kwargs["ExclusiveStartKey"] = last

        df = pd.DataFrame(items)
        log.info(
            "DynamoDB fetch | tenant=%s table=%s items=%d",
            self.tenant_id or "–", table_name, len(df),
        )
        return df
