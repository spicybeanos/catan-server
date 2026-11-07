# Catan Server

FastAPI backend for the base game of Catan. Games are persisted in Postgres as
their replayable action log, so server restarts do not lose state.

## Quick start

```bash
# 1. Start Postgres (matches the default DATABASE_URL)
docker compose up -d

# 2. Install dependencies
uv sync

# 3. Run the API
uv run uvicorn main:app --reload
```

The API is served at <http://127.0.0.1:8000>, with interactive docs at `/docs`.

### Configuration

| Variable       | Default                                          |
| -------------- | ------------------------------------------------ |
| `DATABASE_URL` | `postgresql://catan:catan@localhost:5432/catan` |

See `.env.example`. The schema (`games`, `player_tokens`) is created
automatically on startup.

## API overview

| Method | Path                     | Description                                            |
| ------ | ------------------------ | ------------------------------------------------------ |
| POST   | `/games`                 | Create a game                                          |
| POST   | `/games/{id}/join`       | Join and receive `{player_id, token}`                  |
| POST   | `/games/{id}/start`      | Deal the board (any joined player)                     |
| GET    | `/games/{id}`            | Game state (send `X-Player-Token` to see your hand)    |
| POST   | `/games/{id}/actions`    | `{"action": "roll", "params": {}}`                     |
| GET    | `/games/{id}/placements` | Legal vertices/edges for highlighting                  |
| GET    | `/games/{id}/history`    | Public event feed (`?since=N`)                         |
| GET    | `/games/{id}/log`        | Replayable action log                                  |
| GET    | `/games/meta/actions`    | Available action names                                 |
| GET    | `/health`                | Liveness + database check                              |

## Persistence

Each game is written to the `games` table as its action log (`Game.to_log()`).
On a cache miss the server rebuilds the game deterministically with
`Game.from_log()`. Player tokens are stored in `player_tokens` and cascade-delete
with their game.

The in-process cache is guarded by a per-game lock, which keeps a single server
process consistent. Running multiple API workers against the same games would
need a distributed lock or sticky sessions.
