"""
catan.py - a self-contained rules engine for the base game of Catan (3-4 players).

Design goals
------------
* Pure Python, no web dependencies: import it from FastAPI, a CLI, tests, a bot...
* Every move goes through ``Game.apply(player_id, action, **params)``.  Illegal
  moves raise ``CatanError`` *before* any state is mutated, so a failed request
  never leaves a game half-updated.
* ``Game.to_dict(viewer)`` returns JSON-safe state with other players' hands and
  development cards hidden.
* Games are deterministic given a seed + action log, so persistence is just
  ``game.to_log()`` / ``Game.from_log(data)`` (store the JSON anywhere).

Rules implemented
-----------------
Board (19 hexes, 54 vertices, 72 edges, 9 harbors), snake-order setup with
starting resources, dice production with bank shortages, the robber / 7 / discard
(>7 cards) / steal, building roads/settlements/cities (distance rule, road
connectivity, opponent-blocking), piece limits, development cards (knight, road
building, year of plenty, monopoly, victory point; no playing a card the turn it
was bought; one per turn), longest road (>=5, broken by opponent buildings, ties
keep the holder), largest army (>=3), player trading (propose / accept / confirm),
maritime trading (4:1, 3:1, 2:1), and winning at 10 VP on your own turn.
"""

from __future__ import annotations

import inspect
import math
import random
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Errors & enums
# --------------------------------------------------------------------------- #
class CatanError(Exception):
    """Raised for any illegal move or invalid request."""


class Resource(str, Enum):
    BRICK = "brick"
    LUMBER = "lumber"
    WOOL = "wool"
    GRAIN = "grain"
    ORE = "ore"


class Terrain(str, Enum):
    HILLS = "hills"
    FOREST = "forest"
    PASTURE = "pasture"
    FIELDS = "fields"
    MOUNTAINS = "mountains"
    DESERT = "desert"


class DevCard(str, Enum):
    KNIGHT = "knight"
    VICTORY_POINT = "victory_point"
    ROAD_BUILDING = "road_building"
    YEAR_OF_PLENTY = "year_of_plenty"
    MONOPOLY = "monopoly"


class Phase(str, Enum):
    LOBBY = "lobby"                  # players joining
    SETUP = "setup"                  # initial placement (snake order)
    ROLL = "roll"                    # current player must roll (may play a dev card first)
    DISCARD = "discard"              # a 7 was rolled; players with >7 cards discard
    MOVE_ROBBER = "move_robber"
    STEAL = "steal"                  # choose whom to steal from
    ROAD_BUILDING = "road_building"  # placing free roads from the card
    MAIN = "main"                    # trade / build / play cards / end turn
    FINISHED = "finished"


TERRAIN_RESOURCE = {
    Terrain.HILLS: Resource.BRICK,
    Terrain.FOREST: Resource.LUMBER,
    Terrain.PASTURE: Resource.WOOL,
    Terrain.FIELDS: Resource.GRAIN,
    Terrain.MOUNTAINS: Resource.ORE,
}

COSTS: dict[str, dict[Resource, int]] = {
    "road": {Resource.BRICK: 1, Resource.LUMBER: 1},
    "settlement": {Resource.BRICK: 1, Resource.LUMBER: 1, Resource.WOOL: 1, Resource.GRAIN: 1},
    "city": {Resource.GRAIN: 2, Resource.ORE: 3},
    "dev_card": {Resource.WOOL: 1, Resource.GRAIN: 1, Resource.ORE: 1},
}
PIECE_LIMITS = {"road": 15, "settlement": 5, "city": 4}
DEV_DECK = {
    DevCard.KNIGHT: 14,
    DevCard.VICTORY_POINT: 5,
    DevCard.ROAD_BUILDING: 2,
    DevCard.YEAR_OF_PLENTY: 2,
    DevCard.MONOPOLY: 2,
}
BANK_PER_RESOURCE = 19
HAND_LIMIT = 7
LONGEST_ROAD_MIN = 5
LARGEST_ARMY_MIN = 3
COLORS = ["red", "blue", "orange", "white"]
NUMBER_TOKENS = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
PORT_SLOTS = [0, 3, 6, 10, 13, 16, 20, 23, 26]   # indices into the 30 coastal edges
SQRT3 = math.sqrt(3)
AXIAL_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


# --------------------------------------------------------------------------- #
# Board
# --------------------------------------------------------------------------- #
@dataclass
class Hex:
    id: int
    q: int
    r: int
    terrain: Terrain
    number: Optional[int]


class Board:
    """Hex layout plus the vertex/edge graph. Ids are plain ints and stable."""

    def __init__(self, hexes: list[Hex], rng: random.Random):
        self.hexes = hexes
        self._build_graph()
        self._place_ports(rng)

    # -- generation ---------------------------------------------------------
    @classmethod
    def generate(cls, rng: random.Random) -> "Board":
        coords = [(q, r) for q in range(-2, 3) for r in range(max(-2, -q - 2), min(2, -q + 2) + 1)]
        terrains = (
            [Terrain.FOREST] * 4 + [Terrain.PASTURE] * 4 + [Terrain.FIELDS] * 4
            + [Terrain.HILLS] * 3 + [Terrain.MOUNTAINS] * 3 + [Terrain.DESERT]
        )
        rng.shuffle(terrains)
        numbers = list(NUMBER_TOKENS)
        hexes: list[Hex] = []
        for _ in range(1000):  # re-deal until no two "red" numbers (6/8) touch
            rng.shuffle(numbers)
            it = iter(numbers)
            hexes = [
                Hex(i, q, r, t, None if t == Terrain.DESERT else next(it))
                for i, ((q, r), t) in enumerate(zip(coords, terrains))
            ]
            if not cls._red_adjacent(hexes):
                break
        return cls(hexes, rng)

    @staticmethod
    def _red_adjacent(hexes: list[Hex]) -> bool:
        red = {(h.q, h.r) for h in hexes if h.number in (6, 8)}
        return any((q + dq, r + dr) in red for (q, r) in red for dq, dr in AXIAL_DIRS)

    # -- graph --------------------------------------------------------------
    def _build_graph(self) -> None:
        corners: dict[int, list[tuple[float, float]]] = {}
        points: set[tuple[float, float]] = set()
        for h in self.hexes:
            cx, cy = SQRT3 * (h.q + h.r / 2), 1.5 * h.r
            pts = [
                (round(cx + math.cos(math.radians(60 * i - 30)), 3),
                 round(cy + math.sin(math.radians(60 * i - 30)), 3))
                for i in range(6)
            ]
            corners[h.id] = pts
            points.update(pts)

        ordered = sorted(points, key=lambda p: (p[1], p[0]))
        vid = {p: i for i, p in enumerate(ordered)}
        self.vertices: dict[int, tuple[float, float]] = {i: p for p, i in vid.items()}
        self.hex_vertices = {hid: [vid[p] for p in pts] for hid, pts in corners.items()}
        self.vertex_hexes: dict[int, list[int]] = {v: [] for v in self.vertices}
        for hid, vs in self.hex_vertices.items():
            for v in vs:
                self.vertex_hexes[v].append(hid)

        edge_count: dict[tuple[int, int], int] = {}
        for vs in self.hex_vertices.values():
            for i in range(6):
                key = tuple(sorted((vs[i], vs[(i + 1) % 6])))
                edge_count[key] = edge_count.get(key, 0) + 1

        def mid(key):
            (x1, y1), (x2, y2) = self.vertices[key[0]], self.vertices[key[1]]
            return ((y1 + y2) / 2, (x1 + x2) / 2)

        self.edges: dict[int, tuple[int, int]] = {}
        self.edge_id: dict[tuple[int, int], int] = {}
        for i, key in enumerate(sorted(edge_count, key=mid)):
            self.edges[i] = key
            self.edge_id[key] = i
        self.coastal_edges = [i for i, k in self.edges.items() if edge_count[k] == 1]

        self.vertex_edges: dict[int, list[int]] = {v: [] for v in self.vertices}
        self.vertex_neighbors: dict[int, list[int]] = {v: [] for v in self.vertices}
        for e, (a, b) in self.edges.items():
            self.vertex_edges[a].append(e)
            self.vertex_edges[b].append(e)
            self.vertex_neighbors[a].append(b)
            self.vertex_neighbors[b].append(a)

    def edge_midpoint(self, e: int) -> tuple[float, float]:
        (x1, y1), (x2, y2) = (self.vertices[v] for v in self.edges[e])
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _place_ports(self, rng: random.Random) -> None:
        ring = sorted(self.coastal_edges, key=lambda e: math.atan2(*reversed(self.edge_midpoint(e))))
        types = ["any"] * 4 + [r.value for r in Resource]  # 4x 3:1, 5x 2:1
        rng.shuffle(types)
        self.ports = [
            {"edge": ring[slot], "vertices": list(self.edges[ring[slot]]), "type": t}
            for slot, t in zip(PORT_SLOTS, types)
        ]
        self.vertex_port: dict[int, str] = {v: p["type"] for p in self.ports for v in p["vertices"]}

    def to_dict(self) -> dict:
        return {
            "hexes": [
                {"id": h.id, "q": h.q, "r": h.r, "terrain": h.terrain.value, "number": h.number,
                 "vertices": self.hex_vertices[h.id]}
                for h in self.hexes
            ],
            "vertices": [{"id": v, "x": x, "y": y} for v, (x, y) in self.vertices.items()],
            "edges": [{"id": e, "vertices": list(vs)} for e, vs in self.edges.items()],
            "ports": self.ports,
        }


# --------------------------------------------------------------------------- #
# Player
# --------------------------------------------------------------------------- #
@dataclass
class Player:
    id: str
    name: str
    color: str
    hand: dict[Resource, int] = field(default_factory=lambda: {r: 0 for r in Resource})
    dev_cards: list[DevCard] = field(default_factory=list)          # playable
    dev_cards_new: list[DevCard] = field(default_factory=list)      # bought this turn
    dev_cards_played: list[DevCard] = field(default_factory=list)
    knights_played: int = 0

    def card_count(self) -> int:
        return sum(self.hand.values())

    def has(self, cards: dict[Resource, int]) -> bool:
        return all(self.hand[r] >= n for r, n in cards.items())

    def take(self, cards: dict[Resource, int]) -> None:
        for r, n in cards.items():
            self.hand[r] -= n

    def give(self, cards: dict[Resource, int]) -> None:
        for r, n in cards.items():
            self.hand[r] += n


# --------------------------------------------------------------------------- #
# Game
# --------------------------------------------------------------------------- #
ACTIONS = (
    "roll", "discard", "move_robber", "steal",
    "build_road", "build_settlement", "build_city", "buy_dev_card",
    "play_knight", "play_road_building", "play_year_of_plenty", "play_monopoly",
    "propose_trade", "accept_trade", "reject_trade", "confirm_trade", "cancel_trade",
    "bank_trade", "end_turn",
)


class Game:
    def __init__(
        self,
        seed: Optional[int] = None,
        *,
        min_players: int = 3,
        max_players: int = 4,
        victory_points: int = 10,
        game_id: Optional[str] = None,
    ):
        self.id = game_id or uuid.uuid4().hex[:8]
        self.seed = seed if seed is not None else random.SystemRandom().randrange(2**32)
        self.config = {"min_players": min_players, "max_players": max_players,
                       "victory_points": victory_points}
        self.rng = random.Random(self.seed)

        self.phase = Phase.LOBBY
        self.players: list[Player] = []
        self.board: Optional[Board] = None
        self.bank: dict[Resource, int] = {r: BANK_PER_RESOURCE for r in Resource}
        self.dev_deck: list[DevCard] = []
        self.current = 0                       # index into self.players
        self.turn = 0
        self.dice: Optional[tuple[int, int]] = None
        self.robber: Optional[int] = None

        self.occupancy: dict[int, tuple[str, str]] = {}   # vertex -> (player_id, "settlement"|"city")
        self.road_owner: dict[int, str] = {}               # edge -> player_id

        self.setup_order: list[int] = []
        self.setup_step = 0
        self.setup_stage = "settlement"
        self.setup_settlement: Optional[int] = None

        self.after_robber = Phase.MAIN
        self.resume_phase = Phase.MAIN
        self.discard_pending: dict[str, int] = {}
        self.steal_candidates: list[str] = []
        self.dev_played_this_turn = False
        self.road_building_left = 0
        self.trade: Optional[dict] = None
        self._trade_counter = 0

        self.longest_road_holder: Optional[str] = None
        self.road_lengths: dict[str, int] = {}
        self.largest_army_holder: Optional[str] = None
        self.winner: Optional[str] = None

        self.log: list[dict] = []        # successful actions, for replay
        self.history: list[dict] = []    # public events, for UIs

    # ------------------------------------------------------------------ #
    # Lobby
    # ------------------------------------------------------------------ #
    def add_player(self, name: str, color: Optional[str] = None) -> str:
        self._expect(self.phase == Phase.LOBBY, "The game has already started.")
        self._expect(len(self.players) < self.config["max_players"], "The game is full.")
        self._expect(bool(name and name.strip()), "Name required.")
        taken = {p.color for p in self.players}
        if color is None:
            color = next(c for c in COLORS if c not in taken)
        self._expect(color in COLORS and color not in taken, "Invalid or taken color.")
        player = Player(id=f"p{len(self.players)}", name=name.strip(), color=color)
        self.players.append(player)
        self.log.append({"action": "add_player", "params": {"name": player.name, "color": color}})
        self._emit("join", player.id)
        return player.id

    def start(self) -> None:
        self._expect(self.phase == Phase.LOBBY, "The game has already started.")
        self._expect(len(self.players) >= self.config["min_players"],
                     f"Need at least {self.config['min_players']} players.")
        self.rng.shuffle(self.players)                     # random seating order
        self.board = Board.generate(self.rng)
        self.robber = next(h.id for h in self.board.hexes if h.terrain == Terrain.DESERT)
        self.dev_deck = [c for c, n in DEV_DECK.items() for _ in range(n)]
        self.rng.shuffle(self.dev_deck)
        n = len(self.players)
        self.setup_order = list(range(n)) + list(reversed(range(n)))
        self.setup_step, self.setup_stage, self.current, self.turn = 0, "settlement", 0, 0
        self.phase = Phase.SETUP
        self.log.append({"action": "start", "params": {}})
        self._emit("start", None, order=[p.id for p in self.players])

    # ------------------------------------------------------------------ #
    # Action dispatch
    # ------------------------------------------------------------------ #
    def apply(self, player_id: str, action: str, **params: Any) -> list[dict]:
        """Perform an action. Returns the public events it produced."""
        self._expect(action in ACTIONS, f"Unknown action '{action}'.")
        self._expect(self.phase != Phase.LOBBY, "The game has not started.")
        self._expect(self.phase != Phase.FINISHED, "The game is over.")
        player = self._player(player_id)
        fn = getattr(self, f"_do_{action}")
        try:
            inspect.signature(fn).bind(player, **params)
        except TypeError as exc:
            raise CatanError(f"Invalid parameters for '{action}': {exc}") from None

        first_event = len(self.history)
        fn(player, **params)                 # validates first, then mutates
        self.log.append({"player": player_id, "action": action, "params": params})
        self._check_win()
        return self.history[first_event:]

    def to_log(self) -> dict:
        return {"id": self.id, "seed": self.seed, "config": self.config, "actions": self.log}

    @classmethod
    def from_log(cls, data: dict) -> "Game":
        g = cls(seed=data["seed"], game_id=data.get("id"), **data["config"])
        for e in data["actions"]:
            if e["action"] == "add_player":
                g.add_player(**e["params"])
            elif e["action"] == "start":
                g.start()
            else:
                g.apply(e["player"], e["action"], **e["params"])
        return g

    # ------------------------------------------------------------------ #
    # Actions: dice, robber, discards
    # ------------------------------------------------------------------ #
    def _do_roll(self, p: Player) -> None:
        self._need(p, Phase.ROLL)
        self.dice = (self.rng.randint(1, 6), self.rng.randint(1, 6))
        total = sum(self.dice)
        self._emit("roll", p.id, dice=list(self.dice), total=total)
        if total == 7:
            self.discard_pending = {q.id: q.card_count() // 2
                                    for q in self.players if q.card_count() > HAND_LIMIT}
            self.after_robber = Phase.MAIN
            self.phase = Phase.DISCARD if self.discard_pending else Phase.MOVE_ROBBER
        else:
            self._produce(total)
            self.phase = Phase.MAIN

    def _do_discard(self, p: Player, resources: dict) -> None:
        self._expect(self.phase == Phase.DISCARD, "Not the discard phase.")
        need = self.discard_pending.get(p.id)
        self._expect(need is not None, "You do not need to discard.")
        cards = self._parse_hand(resources)
        self._expect(sum(cards.values()) == need, f"You must discard exactly {need} cards.")
        self._expect(p.has(cards), "You do not have those cards.")
        p.take(cards)
        for r, n in cards.items():
            self.bank[r] += n
        del self.discard_pending[p.id]
        self._emit("discard", p.id, count=need)
        if not self.discard_pending:
            self.phase = Phase.MOVE_ROBBER

    def _do_move_robber(self, p: Player, hex_id: int, victim: Optional[str] = None) -> None:
        self._need(p, Phase.MOVE_ROBBER)
        self._expect(isinstance(hex_id, int) and 0 <= hex_id < len(self.board.hexes), "Unknown hex.")
        self._expect(hex_id != self.robber, "The robber must move to a different hex.")
        candidates = self._steal_candidates(hex_id, p)
        if victim is not None:
            self._expect(victim in candidates, "That player cannot be robbed there.")
        self.robber = hex_id
        self._emit("move_robber", p.id, hex_id=hex_id)
        if not candidates:
            self.phase = self.after_robber
        elif victim is not None or len(candidates) == 1:
            self._steal(p, victim or candidates[0])
        else:
            self.steal_candidates = candidates
            self.phase = Phase.STEAL

    def _do_steal(self, p: Player, victim: str) -> None:
        self._need(p, Phase.STEAL)
        self._expect(victim in self.steal_candidates, "Choose one of the eligible players.")
        self._steal(p, victim)

    def _steal(self, thief: Player, victim_id: str) -> None:
        victim = self._player(victim_id)
        pool = [r for r, n in victim.hand.items() for _ in range(n)]
        card = self.rng.choice(pool)
        victim.hand[card] -= 1
        thief.hand[card] += 1
        self._emit("steal", thief.id, victim=victim_id)
        self.steal_candidates = []
        self.phase = self.after_robber

    # ------------------------------------------------------------------ #
    # Actions: building
    # ------------------------------------------------------------------ #
    def _do_build_settlement(self, p: Player, vertex: int) -> None:
        self._expect(self.phase in (Phase.SETUP, Phase.MAIN), "You cannot build a settlement now.")
        self._expect_current(p)
        self._expect(vertex in self.board.vertices, "Unknown vertex.")
        self._expect(self._free_vertex(vertex), "Too close to another settlement or occupied.")
        self._expect(self._count(p.id, "settlement") < PIECE_LIMITS["settlement"],
                     "No settlements left (upgrade one to a city first).")
        if self.phase == Phase.SETUP:
            self._expect(self.setup_stage == "settlement", "Place your road first.")
        else:
            self._expect(any(self.road_owner.get(e) == p.id for e in self.board.vertex_edges[vertex]),
                         "A settlement must connect to one of your roads.")
            self._pay(p, COSTS["settlement"])
        self.occupancy[vertex] = (p.id, "settlement")
        self._emit("build_settlement", p.id, vertex=vertex)
        if self.phase == Phase.SETUP:
            self.setup_stage, self.setup_settlement = "road", vertex
            if self.setup_step >= len(self.players):         # second round pays out
                for hid in self.board.vertex_hexes[vertex]:
                    res = TERRAIN_RESOURCE.get(self.board.hexes[hid].terrain)
                    if res and self.bank[res] > 0:
                        self.bank[res] -= 1
                        p.hand[res] += 1
        self._update_longest_road()

    def _do_build_road(self, p: Player, edge: int) -> None:
        self._expect(self.phase in (Phase.SETUP, Phase.MAIN, Phase.ROAD_BUILDING),
                     "You cannot build a road now.")
        self._expect_current(p)
        self._expect(edge in self.board.edges, "Unknown edge.")
        self._expect(edge not in self.road_owner, "That edge already has a road.")
        self._expect(self._count(p.id, "road") < PIECE_LIMITS["road"], "No roads left.")
        if self.phase == Phase.SETUP:
            self._expect(self.setup_stage == "road", "Place a settlement first.")
            self._expect(self.setup_settlement in self.board.edges[edge],
                         "The road must touch the settlement you just placed.")
        else:
            self._expect(self._road_connects(p.id, edge), "A road must connect to your network.")
            if self.phase == Phase.MAIN:
                self._pay(p, COSTS["road"])
        self.road_owner[edge] = p.id
        self._emit("build_road", p.id, edge=edge)
        if self.phase == Phase.SETUP:
            self._advance_setup()
        elif self.phase == Phase.ROAD_BUILDING:
            self.road_building_left -= 1
            if self.road_building_left <= 0 or not self._legal_road_edges(p):
                self.road_building_left = 0
                self.phase = self.resume_phase
        self._update_longest_road()

    def _do_build_city(self, p: Player, vertex: int) -> None:
        self._need(p, Phase.MAIN)
        self._expect(self.occupancy.get(vertex) == (p.id, "settlement"),
                     "A city must replace one of your settlements.")
        self._expect(self._count(p.id, "city") < PIECE_LIMITS["city"], "No cities left.")
        self._pay(p, COSTS["city"])
        self.occupancy[vertex] = (p.id, "city")
        self._emit("build_city", p.id, vertex=vertex)

    def _do_buy_dev_card(self, p: Player) -> None:
        self._need(p, Phase.MAIN)
        self._expect(bool(self.dev_deck), "The development card deck is empty.")
        self._pay(p, COSTS["dev_card"])
        p.dev_cards_new.append(self.dev_deck.pop())
        self._emit("buy_dev_card", p.id, deck_remaining=len(self.dev_deck))

    def _advance_setup(self) -> None:
        self.setup_step += 1
        self.setup_settlement = None
        if self.setup_step >= len(self.setup_order):
            self.phase, self.current, self.turn = Phase.ROLL, 0, 1
            self._emit("setup_complete", None)
        else:
            self.current = self.setup_order[self.setup_step]
            self.setup_stage = "settlement"

    # ------------------------------------------------------------------ #
    # Actions: development cards
    # ------------------------------------------------------------------ #
    def _check_dev_playable(self, p: Player, card: DevCard) -> None:
        self._expect_current(p)
        self._expect(self.phase in (Phase.ROLL, Phase.MAIN), "You cannot play a card now.")
        self._expect(not self.dev_played_this_turn, "Only one development card per turn.")
        self._expect(card in p.dev_cards,
                     "You do not have that card (cards bought this turn cannot be played).")

    def _consume_dev(self, p: Player, card: DevCard) -> None:
        p.dev_cards.remove(card)
        p.dev_cards_played.append(card)
        self.dev_played_this_turn = True
        self._emit("play_dev_card", p.id, card=card.value)

    def _do_play_knight(self, p: Player) -> None:
        self._check_dev_playable(p, DevCard.KNIGHT)
        self._consume_dev(p, DevCard.KNIGHT)
        p.knights_played += 1
        self.after_robber = self.phase           # back to ROLL or MAIN afterwards
        self.phase = Phase.MOVE_ROBBER
        self._update_largest_army()

    def _do_play_road_building(self, p: Player) -> None:
        self._check_dev_playable(p, DevCard.ROAD_BUILDING)
        self._expect(self._legal_road_edges(p), "You have no legal road placements.")
        self._consume_dev(p, DevCard.ROAD_BUILDING)
        self.road_building_left = min(2, PIECE_LIMITS["road"] - self._count(p.id, "road"))
        self.resume_phase = self.phase
        self.phase = Phase.ROAD_BUILDING

    def _do_play_year_of_plenty(self, p: Player, resources: list) -> None:
        self._check_dev_playable(p, DevCard.YEAR_OF_PLENTY)
        self._expect(isinstance(resources, (list, tuple)) and len(resources) == 2,
                     "Choose exactly two resources.")
        wanted = Counter(self._resource(r) for r in resources)
        self._expect(all(self.bank[r] >= n for r, n in wanted.items()), "The bank lacks those cards.")
        self._consume_dev(p, DevCard.YEAR_OF_PLENTY)
        for r, n in wanted.items():
            self.bank[r] -= n
            p.hand[r] += n

    def _do_play_monopoly(self, p: Player, resource: str) -> None:
        self._check_dev_playable(p, DevCard.MONOPOLY)
        res = self._resource(resource)
        self._consume_dev(p, DevCard.MONOPOLY)
        total = 0
        for q in self.players:
            if q.id != p.id:
                total += q.hand[res]
                p.hand[res] += q.hand[res]
                q.hand[res] = 0
        self._emit("monopoly", p.id, resource=res.value, count=total)

    # ------------------------------------------------------------------ #
    # Actions: trading
    # ------------------------------------------------------------------ #
    def _do_propose_trade(self, p: Player, give: dict, get: dict, to: Optional[list] = None) -> None:
        self._need(p, Phase.MAIN)
        give, get = self._parse_hand(give), self._parse_hand(get)
        self._expect(give and get, "A trade must offer and request cards.")
        self._expect(not (set(give) & set(get)), "You cannot trade a resource for the same resource.")
        self._expect(p.has(give), "You do not have the offered cards.")
        others = [q.id for q in self.players if q.id != p.id]
        to = others if to is None else list(to)
        self._expect(to and all(t in others for t in to), "Invalid trade recipients.")
        self._trade_counter += 1
        self.trade = {"id": self._trade_counter, "proposer": p.id, "give": give, "get": get,
                      "to": to, "accepted": [], "rejected": []}
        self._emit("propose_trade", p.id, trade_id=self._trade_counter)

    def _do_accept_trade(self, p: Player) -> None:
        t = self._open_trade_for(p)
        self._expect(p.has(t["get"]), "You do not have the requested cards.")
        if p.id not in t["accepted"]:
            t["accepted"].append(p.id)
        if p.id in t["rejected"]:
            t["rejected"].remove(p.id)
        self._emit("accept_trade", p.id)

    def _do_reject_trade(self, p: Player) -> None:
        t = self._open_trade_for(p)
        if p.id in t["accepted"]:
            t["accepted"].remove(p.id)
        if p.id not in t["rejected"]:
            t["rejected"].append(p.id)
        self._emit("reject_trade", p.id)

    def _do_confirm_trade(self, p: Player, partner: str) -> None:
        self._need(p, Phase.MAIN)
        t = self.trade
        self._expect(t and t["proposer"] == p.id, "You have no open trade.")
        self._expect(partner in t["accepted"], "That player has not accepted.")
        other = self._player(partner)
        self._expect(p.has(t["give"]) and other.has(t["get"]), "Someone no longer has the cards.")
        p.take(t["give"]); other.give(t["give"])
        other.take(t["get"]); p.give(t["get"])
        self.trade = None
        self._emit("trade", p.id, partner=partner, give=self._hand_out(t["give"]), get=self._hand_out(t["get"]))

    def _do_cancel_trade(self, p: Player) -> None:
        self._need(p, Phase.MAIN)
        self._expect(self.trade and self.trade["proposer"] == p.id, "You have no open trade.")
        self.trade = None
        self._emit("cancel_trade", p.id)

    def _do_bank_trade(self, p: Player, give: str, get: str, count: int = 1) -> None:
        self._need(p, Phase.MAIN)
        give, get = self._resource(give), self._resource(get)
        self._expect(give != get, "Choose two different resources.")
        self._expect(isinstance(count, int) and count >= 1, "Invalid count.")
        ratio = self.trade_ratio(p.id, give)
        self._expect(p.hand[give] >= ratio * count, f"You need {ratio * count} {give.value}.")
        self._expect(self.bank[get] >= count, "The bank lacks that resource.")
        p.hand[give] -= ratio * count
        self.bank[give] += ratio * count
        self.bank[get] -= count
        p.hand[get] += count
        self._emit("bank_trade", p.id, give=give.value, get=get.value, ratio=ratio, count=count)

    def _do_end_turn(self, p: Player) -> None:
        self._need(p, Phase.MAIN)
        p.dev_cards += p.dev_cards_new
        p.dev_cards_new = []
        self.trade = None
        self.dev_played_this_turn = False
        self.dice = None
        self.current = (self.current + 1) % len(self.players)
        self.turn += 1
        self.phase = Phase.ROLL
        self._emit("end_turn", p.id, next=self.players[self.current].id)

    # ------------------------------------------------------------------ #
    # Queries (safe to call any time)
    # ------------------------------------------------------------------ #
    @property
    def current_player(self) -> Player:
        return self.players[self.current]

    def trade_ratio(self, player_id: str, resource: Resource | str) -> int:
        res = self._resource(resource)
        ratio = 4
        for v, (owner, _) in self.occupancy.items():
            if owner != player_id:
                continue
            port = self.board.vertex_port.get(v)
            if port == "any":
                ratio = min(ratio, 3)
            elif port == res.value:
                ratio = min(ratio, 2)
        return ratio

    def victory_points(self, player: Player, include_hidden: bool = True) -> int:
        vp = self._count(player.id, "settlement") + 2 * self._count(player.id, "city")
        vp += 2 * (self.longest_road_holder == player.id) + 2 * (self.largest_army_holder == player.id)
        if include_hidden:
            vp += (player.dev_cards + player.dev_cards_new).count(DevCard.VICTORY_POINT)
        return vp

    def placement_options(self, player_id: str) -> dict[str, list[int]]:
        """Legal vertices/edges for the player right now (ignores resource cost)."""
        p = self._player(player_id)
        out: dict[str, list[int]] = {"settlements": [], "cities": [], "roads": []}
        if self.current_player.id != p.id:
            return out
        if self.phase == Phase.SETUP:
            if self.setup_stage == "settlement":
                out["settlements"] = [v for v in self.board.vertices if self._free_vertex(v)]
            else:
                out["roads"] = [e for e in self.board.vertex_edges[self.setup_settlement]
                                if e not in self.road_owner]
        elif self.phase in (Phase.MAIN, Phase.ROAD_BUILDING):
            out["roads"] = self._legal_road_edges(p)
            if self.phase == Phase.MAIN:
                out["settlements"] = self._legal_settlement_vertices(p)
                if self._count(p.id, "city") < PIECE_LIMITS["city"]:
                    out["cities"] = self._buildings(p.id, "settlement")
        return out

    def available_actions(self, player_id: str) -> list[str]:
        p = self._player(player_id)
        if self.phase in (Phase.LOBBY, Phase.FINISHED):
            return []
        if self.phase == Phase.DISCARD:
            return ["discard"] if p.id in self.discard_pending else []
        acts: list[str] = []
        if self.trade and p.id in self.trade["to"] and self.phase == Phase.MAIN:
            acts += ["accept_trade", "reject_trade"]
        if self.current_player.id != p.id:
            return acts
        opts = self.placement_options(p.id)
        ph = self.phase
        if ph == Phase.SETUP:
            return ["build_settlement" if self.setup_stage == "settlement" else "build_road"]
        if ph == Phase.MOVE_ROBBER:
            return ["move_robber"]
        if ph == Phase.STEAL:
            return ["steal"]
        if ph == Phase.ROAD_BUILDING:
            return ["build_road"]
        if ph == Phase.ROLL:
            acts.append("roll")
        if ph == Phase.MAIN:
            if p.has(COSTS["road"]) and opts["roads"]:
                acts.append("build_road")
            if p.has(COSTS["settlement"]) and opts["settlements"]:
                acts.append("build_settlement")
            if p.has(COSTS["city"]) and opts["cities"]:
                acts.append("build_city")
            if p.has(COSTS["dev_card"]) and self.dev_deck:
                acts.append("buy_dev_card")
            if p.card_count():
                acts.append("propose_trade")
            if any(p.hand[r] >= self.trade_ratio(p.id, r) for r in Resource):
                acts.append("bank_trade")
            if self.trade:
                acts += ["confirm_trade", "cancel_trade"] if self.trade["accepted"] else ["cancel_trade"]
            acts.append("end_turn")
        if ph in (Phase.ROLL, Phase.MAIN) and not self.dev_played_this_turn:
            playable = set(p.dev_cards)
            if DevCard.KNIGHT in playable:
                acts.append("play_knight")
            if DevCard.ROAD_BUILDING in playable and self._legal_road_edges(p):
                acts.append("play_road_building")
            if DevCard.YEAR_OF_PLENTY in playable:
                acts.append("play_year_of_plenty")
            if DevCard.MONOPOLY in playable:
                acts.append("play_monopoly")
        return acts

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self, viewer: Optional[str] = None) -> dict:
        """JSON-safe snapshot. ``viewer`` is a player id (or None for spectators)."""
        finished = self.phase == Phase.FINISHED
        players = []
        for p in self.players:
            d = {
                "id": p.id, "name": p.name, "color": p.color,
                "resource_cards": p.card_count(),
                "dev_cards": len(p.dev_cards) + len(p.dev_cards_new),
                "dev_cards_played": [c.value for c in p.dev_cards_played],
                "knights_played": p.knights_played,
                "settlements": self._buildings(p.id, "settlement"),
                "cities": self._buildings(p.id, "city"),
                "roads": [e for e, o in self.road_owner.items() if o == p.id],
                "longest_road_length": self.road_lengths.get(p.id, 0),
                "has_longest_road": self.longest_road_holder == p.id,
                "has_largest_army": self.largest_army_holder == p.id,
                "public_victory_points": self.victory_points(p, include_hidden=False),
            }
            if viewer == p.id or finished:
                d["victory_points"] = self.victory_points(p)
            if viewer == p.id:
                d["hand"] = self._hand_out(p.hand)
                d["dev_cards_playable"] = [c.value for c in p.dev_cards]
                d["dev_cards_new"] = [c.value for c in p.dev_cards_new]
            players.append(d)

        trade = None
        if self.trade:
            t = self.trade
            trade = {**t, "give": self._hand_out(t["give"]), "get": self._hand_out(t["get"])}
        return {
            "id": self.id,
            "phase": self.phase.value,
            "turn": self.turn,
            "config": self.config,
            "current_player": self.current_player.id if self.players and self.phase != Phase.LOBBY else None,
            "players": players,
            "board": self.board.to_dict() if self.board else None,
            "robber": self.robber,
            "dice": list(self.dice) if self.dice else None,
            "bank": self._hand_out(self.bank),
            "dev_deck_remaining": len(self.dev_deck),
            "setup": {"step": self.setup_step, "stage": self.setup_stage,
                      "settlement": self.setup_settlement} if self.phase == Phase.SETUP else None,
            "discard_pending": dict(self.discard_pending),
            "steal_candidates": list(self.steal_candidates),
            "road_building_left": self.road_building_left,
            "longest_road_holder": self.longest_road_holder,
            "largest_army_holder": self.largest_army_holder,
            "trade": trade,
            "winner": self.winner,
            "viewer": viewer,
            "available_actions": self.available_actions(viewer) if viewer else [],
        }

    def history_since(self, index: int = 0) -> list[dict]:
        return self.history[index:]

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _expect(cond: Any, msg: str) -> None:
        if not cond:
            raise CatanError(msg)

    def _expect_current(self, p: Player) -> None:
        self._expect(self.current_player.id == p.id, "It is not your turn.")

    def _need(self, p: Player, phase: Phase) -> None:
        self._expect(self.phase == phase, f"You cannot do that during the '{self.phase.value}' phase.")
        self._expect_current(p)

    def _player(self, pid: str) -> Player:
        for p in self.players:
            if p.id == pid:
                return p
        raise CatanError(f"Unknown player '{pid}'.")

    def _emit(self, type_: str, player: Optional[str], **data: Any) -> None:
        self.history.append({"n": len(self.history), "turn": self.turn, "type": type_,
                             "player": player, **data})

    @staticmethod
    def _resource(value: Any) -> Resource:
        try:
            return Resource(value)
        except ValueError:
            raise CatanError(f"Unknown resource '{value}'.") from None

    def _parse_hand(self, raw: Any) -> dict[Resource, int]:
        self._expect(isinstance(raw, dict), "Resources must be an object like {\"brick\": 1}.")
        out: dict[Resource, int] = {}
        for k, v in raw.items():
            self._expect(isinstance(v, int) and not isinstance(v, bool) and v >= 0, "Invalid amount.")
            if v:
                r = self._resource(k)
                out[r] = out.get(r, 0) + v
        return out

    @staticmethod
    def _hand_out(h: dict[Resource, int]) -> dict[str, int]:
        return {r.value: n for r, n in h.items()}

    def _pay(self, p: Player, cost: dict[Resource, int]) -> None:
        self._expect(p.has(cost), "You cannot afford that.")
        p.take(cost)
        for r, n in cost.items():
            self.bank[r] += n

    def _buildings(self, pid: str, kind: str) -> list[int]:
        return [v for v, (o, k) in self.occupancy.items() if o == pid and k == kind]

    def _count(self, pid: str, kind: str) -> int:
        if kind == "road":
            return sum(1 for o in self.road_owner.values() if o == pid)
        return len(self._buildings(pid, kind))

    def _free_vertex(self, v: int) -> bool:
        """Unoccupied and obeying the distance rule (no adjacent building)."""
        return v not in self.occupancy and all(n not in self.occupancy
                                               for n in self.board.vertex_neighbors[v])

    def _legal_settlement_vertices(self, p: Player) -> list[int]:
        if self._count(p.id, "settlement") >= PIECE_LIMITS["settlement"]:
            return []
        return [v for v in self.board.vertices if self._free_vertex(v)
                and any(self.road_owner.get(e) == p.id for e in self.board.vertex_edges[v])]

    def _road_connects(self, pid: str, edge: int) -> bool:
        for v in self.board.edges[edge]:
            occ = self.occupancy.get(v)
            if occ and occ[0] != pid:
                continue                       # opponents' buildings block the connection
            if occ:
                return True
            if any(self.road_owner.get(e) == pid for e in self.board.vertex_edges[v] if e != edge):
                return True
        return False

    def _legal_road_edges(self, p: Player) -> list[int]:
        if self._count(p.id, "road") >= PIECE_LIMITS["road"]:
            return []
        return [e for e in self.board.edges if e not in self.road_owner and self._road_connects(p.id, e)]

    def _steal_candidates(self, hex_id: int, thief: Player) -> list[str]:
        out: list[str] = []
        for v in self.board.hex_vertices[hex_id]:
            occ = self.occupancy.get(v)
            if occ and occ[0] != thief.id and occ[0] not in out and self._player(occ[0]).card_count() > 0:
                out.append(occ[0])
        return out

    def _produce(self, total: int) -> None:
        owed: dict[Resource, dict[str, int]] = {}
        for h in self.board.hexes:
            if h.number != total or h.id == self.robber:
                continue
            res = TERRAIN_RESOURCE[h.terrain]
            for v in self.board.hex_vertices[h.id]:
                occ = self.occupancy.get(v)
                if occ:
                    amt = 2 if occ[1] == "city" else 1
                    owed.setdefault(res, {}).setdefault(occ[0], 0)
                    owed[res][occ[0]] += amt
        gains: dict[str, dict[str, int]] = {}
        for res, by_player in owed.items():
            if self.bank[res] >= sum(by_player.values()):
                grant = dict(by_player)
            elif len(by_player) == 1:                      # only one claimant: gets what is left
                pid, amt = next(iter(by_player.items()))
                grant = {pid: min(amt, self.bank[res])}
            else:                                          # shortage with several claimants: nobody
                grant = {}
            for pid, amt in grant.items():
                if amt:
                    self._player(pid).hand[res] += amt
                    self.bank[res] -= amt
                    gains.setdefault(pid, {})[res.value] = amt
        self._emit("produce", None, gains=gains)

    # -- longest road / largest army / victory --------------------------------
    def _longest_road_for(self, pid: str) -> int:
        adj: dict[int, list[tuple[int, int]]] = {}
        for e, owner in self.road_owner.items():
            if owner == pid:
                a, b = self.board.edges[e]
                adj.setdefault(a, []).append((e, b))
                adj.setdefault(b, []).append((e, a))

        def blocked(v: int) -> bool:
            occ = self.occupancy.get(v)
            return bool(occ and occ[0] != pid)

        def dfs(v: int, used: set[int]) -> int:
            best = len(used)
            for e, w in adj.get(v, []):
                if e in used:
                    continue
                used.add(e)
                best = max(best, len(used) if blocked(w) else dfs(w, used))
                used.remove(e)
            return best

        return max((dfs(v, set()) for v in adj), default=0)

    def _update_longest_road(self) -> None:
        self.road_lengths = {p.id: self._longest_road_for(p.id) for p in self.players}
        best = max(self.road_lengths.values(), default=0)
        if best < LONGEST_ROAD_MIN:
            new = None
        else:
            leaders = [pid for pid, n in self.road_lengths.items() if n == best]
            if self.longest_road_holder in leaders:
                new = self.longest_road_holder            # ties keep the current holder
            else:
                new = leaders[0] if len(leaders) == 1 else None
        if new != self.longest_road_holder:
            self.longest_road_holder = new
            self._emit("longest_road", new, length=best)

    def _update_largest_army(self) -> None:
        holder = self._player(self.largest_army_holder) if self.largest_army_holder else None
        for p in self.players:
            if p.knights_played >= LARGEST_ARMY_MIN and (holder is None or p.knights_played > holder.knights_played):
                holder = p
        new = holder.id if holder else None
        if new != self.largest_army_holder:
            self.largest_army_holder = new
            self._emit("largest_army", new)

    def _check_win(self) -> None:
        if self.phase in (Phase.LOBBY, Phase.SETUP, Phase.FINISHED):
            return
        cur = self.current_player          # you can only win on your own turn
        if self.victory_points(cur) >= self.config["victory_points"]:
            self.winner = cur.id
            self.phase = Phase.FINISHED
            self._emit("win", cur.id)
