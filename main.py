"""
main.py - FastAPI app for the Catan rules engine in catan.py.

Run:   uv sync
       docker compose up -d          # start Postgres (or point DATABASE_URL at one)
       uvicorn main:app --reload

Flow:  POST /games                      -> create a game
       POST /games/{id}/join            -> returns {player_id, token}
       POST /games/{id}/start           -> (any joined player) deals the board
       GET  /games/{id}                 -> state (send X-Player-Token to see your hand)
       POST /games/{id}/actions         -> {"action": "roll", "params": {}}
       GET  /games/{id}/placements      -> legal vertices / edges for highlighting
       GET  /games/{id}/history?since=N -> event feed (poll this, or add a WebSocket)
       GET  /games/{id}/log             -> replayable log

Persistence: every game is stored in Postgres as its replayable action log and
rebuilt with ``Game.from_log``. Player tokens live in the ``player_tokens``
table. Set ``DATABASE_URL`` (defaults to the docker-compose instance):
``postgresql://catan:catan@localhost:5432/catan``.
"""

from __future__ import annotations

import logging
import secrets
import threading
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import APIRouter, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from catan import ACTIONS, CatanError, Game
from db import GameStore

log = logging.getLogger(__name__)

router = APIRouter(prefix="/games", tags=["catan"])
store = GameStore()


# --------------------------------------------------------------------------- #
# Storage (Postgres-backed, with a per-process cache for active games)
# --------------------------------------------------------------------------- #
class Entry:
    def __init__(self, game: Game, tokens: Optional[dict[str, str]] = None):
        self.game = game
        self.lock = threading.RLock()                 # sync endpoints run in a thread pool
        self.tokens = tokens or {}                    # token -> player_id


GAMES: dict[str, Entry] = {}
GAMES_LOCK = threading.Lock()


def get_entry(game_id: str) -> Entry:
    entry = GAMES.get(game_id)
    if entry:
        return entry
    with GAMES_LOCK:
        entry = GAMES.get(game_id)
        if entry:
            return entry
        try:
            game = store.load_game(game_id)
            tokens = store.load_tokens(game_id) if game else {}
        except Exception as exc:
            log.exception("Failed to load game %s", game_id)
            raise HTTPException(503, "Could not load game state") from exc
        if game is None:
            raise HTTPException(404, "Game not found")
        entry = Entry(game, tokens)
        GAMES[game_id] = entry
        return entry


def persist(entry: Entry, token: Optional[tuple[str, str]] = None) -> None:
    """Write the game (and optionally a new token) to Postgres."""
    try:
        store.save_game(entry.game, token=token)
    except Exception as exc:
        log.exception("Failed to persist game %s", entry.game.id)
        if GAMES.get(entry.game.id) is entry:     # force a reload from the last good state
            GAMES.pop(entry.game.id, None)
        raise HTTPException(503, "Could not persist game state") from exc


def player_from_token(entry: Entry, token: Optional[str], required: bool = True) -> Optional[str]:
    pid = entry.tokens.get(token or "")
    if required and not pid:
        raise HTTPException(401, "Missing or invalid X-Player-Token")
    return pid


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class CreateGame(BaseModel):
    seed: Optional[int] = None
    victory_points: int = Field(10, ge=3, le=20)
    min_players: int = Field(3, ge=2, le=4)
    max_players: int = Field(4, ge=2, le=4)


class JoinGame(BaseModel):
    name: str = Field(..., min_length=1, max_length=30)
    color: Optional[str] = None


class ActionRequest(BaseModel):
    action: str
    params: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Routes (plain `def` so blocking engine calls run in FastAPI's thread pool)
# --------------------------------------------------------------------------- #
@router.post("", status_code=201)
def create_game(body: CreateGame):
    game = Game(seed=body.seed, victory_points=body.victory_points,
                min_players=body.min_players, max_players=body.max_players)
    entry = Entry(game)
    persist(entry)
    GAMES[game.id] = entry
    return {"game_id": game.id}


@router.post("/{game_id}/join", status_code=201)
def join_game(game_id: str, body: JoinGame):
    entry = get_entry(game_id)
    with entry.lock:
        try:
            pid = entry.game.add_player(body.name, body.color)
        except CatanError as e:
            raise HTTPException(400, str(e))
        token = secrets.token_urlsafe(16)
        persist(entry, token=(token, pid))
        entry.tokens[token] = pid
    return {"player_id": pid, "token": token}


@router.post("/{game_id}/start")
def start_game(game_id: str, x_player_token: Optional[str] = Header(None)):
    entry = get_entry(game_id)
    player_from_token(entry, x_player_token)
    with entry.lock:
        try:
            entry.game.start()
        except CatanError as e:
            raise HTTPException(400, str(e))
        persist(entry)
        return entry.game.to_dict(entry.tokens[x_player_token])


@router.get("/{game_id}")
def get_state(game_id: str, x_player_token: Optional[str] = Header(None)):
    entry = get_entry(game_id)
    viewer = player_from_token(entry, x_player_token, required=False)
    with entry.lock:
        return entry.game.to_dict(viewer)


@router.post("/{game_id}/actions")
def do_action(game_id: str, body: ActionRequest, x_player_token: Optional[str] = Header(None)):
    entry = get_entry(game_id)
    pid = player_from_token(entry, x_player_token)
    with entry.lock:
        try:
            events = entry.game.apply(pid, body.action, **body.params)
        except CatanError as e:
            raise HTTPException(400, str(e))
        persist(entry)
        return {"events": events, "state": entry.game.to_dict(pid)}


@router.get("/{game_id}/placements")
def placements(game_id: str, x_player_token: Optional[str] = Header(None)):
    entry = get_entry(game_id)
    pid = player_from_token(entry, x_player_token)
    with entry.lock:
        return entry.game.placement_options(pid)


@router.get("/{game_id}/history")
def history(game_id: str, since: int = 0):
    entry = get_entry(game_id)
    with entry.lock:
        return entry.game.history_since(since)


@router.get("/{game_id}/log")
def replay_log(game_id: str):
    entry = get_entry(game_id)
    with entry.lock:
        return entry.game.to_log()


@router.get("/meta/actions")
def list_actions():
    return {"actions": list(ACTIONS)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.open()
    try:
        yield
    finally:
        store.close()


app = FastAPI(title="Catan Server", lifespan=lifespan)


@app.get("/health")
def health():
    try:
        store.ping()
    except Exception as exc:
        raise HTTPException(503, "Database unavailable") from exc
    return {"status": "ok"}


app.include_router(router)
