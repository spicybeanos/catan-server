"""
db.py - Postgres persistence for the Catan server.

A game is stored as its replayable action log (``Game.to_log()``), so the full
state can be rebuilt deterministically with ``Game.from_log()``.  Player tokens
live in their own table and are loaded alongside the game.

Configure the connection with ``DATABASE_URL``::

    postgresql://catan:catan@localhost:5432/catan
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolTimeout

from catan import Game

load_dotenv()

log = logging.getLogger(__name__)

DEFAULT_DATABASE_URL = "postgresql://catan:catan@localhost:5432/catan"

SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS games (
        id          TEXT PRIMARY KEY,
        seed        BIGINT NOT NULL,
        config      JSONB NOT NULL,
        log         JSONB NOT NULL DEFAULT '[]'::jsonb,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS player_tokens (
        token       TEXT PRIMARY KEY,
        game_id     TEXT NOT NULL REFERENCES games(id) ON DELETE CASCADE,
        player_id   TEXT NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS player_tokens_game_id_idx
        ON player_tokens (game_id)
    """,
)

UPSERT_GAME = """
    INSERT INTO games (id, seed, config, log)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE
        SET seed = EXCLUDED.seed,
            config = EXCLUDED.config,
            log = EXCLUDED.log,
            updated_at = now()
"""


class GameStore:
    """Small synchronous repository around a psycopg connection pool."""

    def __init__(
        self,
        database_url: Optional[str] = None,
        *,
        min_size: int = 1,
        max_size: int = 10,
        connect_timeout: float = 10.0,
    ):
        self.database_url = database_url or os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
        self.connect_timeout = connect_timeout
        self._min_size = min_size
        self._max_size = max_size
        self._opened = False
        self.pool = self._new_pool()

    def _new_pool(self) -> ConnectionPool:
        return ConnectionPool(
            self.database_url,
            min_size=self._min_size,
            max_size=self._max_size,
            open=False,
            timeout=self.connect_timeout,
            kwargs={"row_factory": dict_row},
        )

    # -- lifecycle ----------------------------------------------------------
    def open(self) -> None:
        if self._opened:
            return
        if self.pool.closed:
            self.pool = self._new_pool()
        try:
            self.pool.open(wait=True, timeout=self.connect_timeout)
        except PoolTimeout as exc:
            self.pool.close()
            raise RuntimeError(
                "Could not connect to Postgres. Is the database running and is "
                "DATABASE_URL set correctly?"
            ) from exc
        try:
            self._init_schema()
        except Exception:
            self.pool.close()
            raise
        self._opened = True

    def close(self) -> None:
        if self._opened:
            self.pool.close()
            self._opened = False

    def _init_schema(self) -> None:
        with self.pool.connection() as conn:
            for statement in SCHEMA:
                conn.execute(statement)

    def ping(self) -> None:
        with self.pool.connection() as conn:
            conn.execute("SELECT 1")

    # -- games --------------------------------------------------------------
    def save_game(self, game: Game, token: Optional[tuple[str, str]] = None) -> None:
        """Insert or update a game, optionally adding a player token atomically."""
        with self.pool.connection() as conn:
            conn.execute(
                UPSERT_GAME,
                (game.id, game.seed, Jsonb(game.config), Jsonb(game.log)),
            )
            if token is not None:
                token_value, player_id = token
                conn.execute(
                    """
                    INSERT INTO player_tokens (token, game_id, player_id)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (token) DO NOTHING
                    """,
                    (token_value, game.id, player_id),
                )

    def load_game(self, game_id: str) -> Optional[Game]:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT seed, config, log FROM games WHERE id = %s",
                (game_id,),
            ).fetchone()
        if row is None:
            return None
        config = row["config"]
        actions = row["log"]
        if isinstance(config, str):
            config = json.loads(config)
        if isinstance(actions, str):
            actions = json.loads(actions)
        return Game.from_log(
            {"id": game_id, "seed": row["seed"], "config": config, "actions": actions}
        )

    def delete_game(self, game_id: str) -> bool:
        with self.pool.connection() as conn:
            result = conn.execute("DELETE FROM games WHERE id = %s", (game_id,))
        return result.rowcount > 0

    # -- tokens -------------------------------------------------------------
    def load_tokens(self, game_id: str) -> dict[str, str]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT token, player_id FROM player_tokens WHERE game_id = %s",
                (game_id,),
            ).fetchall()
        return {row["token"]: row["player_id"] for row in rows}
