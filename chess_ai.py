#!/usr/bin/env python3
"""
CHessai - Local AI vs AI chess.

A single AI (default: Mistral running locally via Ollama) plays BOTH sides.
Every move the model proposes is validated with python-chess; if it is illegal
or unparseable the model is prompted again (with feedback) until it returns a
legal move, or a random legal move is played as a last resort.

You can switch models or point at any OpenAI-compatible API (with an API key)
via config.json or command-line flags.  See README.md.
"""

import argparse
import json
import os
import random
import re
import sys
import time

try:
    import chess
    import chess.pgn
except ImportError:
    sys.exit("Missing dependency 'python-chess'. Install it with:  pip install chess")

try:
    import requests
except ImportError:
    sys.exit("Missing dependency 'requests'. Install it with:  pip install requests")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "provider": "ollama",                       # "ollama" or "openai"
    "model": "mistral",                         # any model your provider knows
    "ollama_host": "http://localhost:11434",    # used when provider == "ollama"
    "base_url": "https://api.openai.com/v1",     # used when provider == "openai"
    "api_key": "",                              # openai provider; or OPENAI_API_KEY env
    "temperature": 0.7,
    "max_retries": 5,                           # re-prompts before random fallback
    "fallback_random": True,                    # play a random legal move if AI fails
    "move_delay": 0.8,                          # seconds to pause between moves
    "max_moves": 200,                           # safety cap, in plies (half-moves)
    "unicode_board": False,                     # True for ♟ glyphs, False for letters
    "request_timeout": 120,                     # seconds per model call
}

UNICODE_PIECES = {
    "K": "♔", "Q": "♕", "R": "♖", "B": "♗", "N": "♘", "P": "♙",
    "k": "♚", "q": "♛", "r": "♜", "b": "♝", "n": "♞", "p": "♟",
}

UCI_RE = re.compile(r"[a-h][1-8][a-h][1-8][qrbnQRBN]?")
SAN_RE = re.compile(r"O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?")


# --------------------------------------------------------------------------- #
# LLM client (Ollama native API or any OpenAI-compatible API)
# --------------------------------------------------------------------------- #
class LLMClient:
    def __init__(self, cfg):
        self.cfg = cfg

    def chat(self, messages):
        if self.cfg["provider"] == "ollama":
            return self._ollama_chat(messages)
        return self._openai_chat(messages)

    def _ollama_chat(self, messages):
        url = self.cfg["ollama_host"].rstrip("/") + "/api/chat"
        payload = {
            "model": self.cfg["model"],
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.cfg["temperature"]},
        }
        r = requests.post(url, json=payload, timeout=self.cfg["request_timeout"])
        r.raise_for_status()
        return r.json()["message"]["content"]

    def _openai_chat(self, messages):
        url = self.cfg["base_url"].rstrip("/") + "/chat/completions"
        key = self.cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": self.cfg["model"],
            "messages": messages,
            "temperature": self.cfg["temperature"],
        }
        r = requests.post(url, json=payload, headers=headers,
                          timeout=self.cfg["request_timeout"])
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# Move parsing & legality
# --------------------------------------------------------------------------- #
def _try_uci(token, board):
    try:
        move = chess.Move.from_uci(token.lower())
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def _try_san(token, board):
    try:
        return board.parse_san(token)   # parse_san only returns legal moves
    except ValueError:
        return None


def extract_move(text, board):
    """Pull a single *legal* move out of the model's reply, or return None."""
    if not text:
        return None
    cleaned = text.strip().strip(".`'\"")

    # 1) The whole reply is just the move (the well-behaved case).
    for parser in (_try_uci, _try_san):
        move = parser(cleaned, board)
        if move:
            return move

    # 2) Hunt for a UCI-looking token anywhere in the text.
    for token in UCI_RE.findall(cleaned):
        move = _try_uci(token, board)
        if move:
            return move

    # 3) Fall back to SAN-looking tokens (e4, Nf3, O-O, exd5, e8=Q ...).
    for token in SAN_RE.findall(cleaned):
        move = _try_san(token, board)
        if move:
            return move

    return None


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "You are a chess engine AND a lively commentator, playing a full game by "
    "yourself and choosing the best move for whichever side is to move. "
    "Respond in EXACTLY this two-line format and nothing else:\n"
    "MOVE: <one legal move in UCI long algebraic notation, e.g. e2e4, g1f3, "
    "e1g1 for castling, or e7e8q for promotion>\n"
    "COMMENT: <one or two short, lively sentences, in the first person, about "
    "your plan or why you played it>"
)


def build_messages(board, retry_feedback=None):
    color = "White" if board.turn == chess.WHITE else "Black"
    legal = " ".join(sorted(m.uci() for m in board.legal_moves))
    user = (
        f"You are playing as {color}.\n"
        f"FEN: {board.fen()}\n\n"
        f"Board (rank 8 = top, uppercase = White):\n{render_board(board, {'unicode_board': False})}\n\n"
        f"Legal moves (UCI): {legal}\n\n"
        "Pick your move from the legal-moves list and reply using the "
        "MOVE/COMMENT format."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if retry_feedback:
        messages.append({"role": "user", "content": retry_feedback})
    return messages


def parse_reply(text, board):
    """Split a model reply into (legal_move_or_None, comment_string)."""
    if not text:
        return None, ""

    move = None
    move_line = re.search(r"move\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if move_line:
        move = extract_move(move_line.group(1), board)
    if move is None:                       # be lenient: scan the whole reply
        move = extract_move(text, board)

    comment = ""
    comment_line = re.search(r"comment\s*[:=]\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if comment_line:
        comment = " ".join(comment_line.group(1).split()).strip()
    return move, comment


def get_ai_move(client, board, cfg):
    """Ask the model for a move + commentary, re-prompting until the move is legal.

    Returns (move, comment, attempts). Raises ConnectionError if the model could
    not be reached at all, or RuntimeError if it never produced a legal move and
    random fallback is disabled.
    """
    last_bad = None
    got_response = False
    net_error = None
    for attempt in range(1, cfg["max_retries"] + 1):
        feedback = None
        if last_bad is not None:
            legal = " ".join(sorted(m.uci() for m in board.legal_moves))
            feedback = (
                f"'{last_bad}' is NOT a legal move here. Choose ONE move from this "
                f"exact list and keep the MOVE/COMMENT format: {legal}"
            )
        try:
            raw = client.chat(build_messages(board, feedback))
        except requests.RequestException as exc:
            net_error = exc
            print(f"    [network error: {exc}] retrying...")
            time.sleep(1.5)
            continue

        got_response = True
        raw = (raw or "").strip()
        move, comment = parse_reply(raw, board)
        if move:
            return move, comment, attempt
        preview = raw.replace("\n", " ")[:60] or "(empty reply)"
        print(f"    attempt {attempt}/{cfg['max_retries']}: illegal/unparseable -> {preview!r}")
        last_bad = preview[:40]

    if not got_response and net_error is not None:
        raise ConnectionError(str(net_error))

    if cfg["fallback_random"]:
        move = random.choice(list(board.legal_moves))
        print(f"    [fallback] AI failed {cfg['max_retries']}x; playing random legal move {move.uci()}")
        return (move,
                "(The model didn't return a legal move, so I'm playing a random legal one.)",
                cfg["max_retries"])

    raise RuntimeError("AI could not produce a legal move and fallback is disabled.")


# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #
def render_board(board, cfg):
    use_unicode = cfg.get("unicode_board", False)
    lines = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            if piece is None:
                cells.append(".")
            else:
                sym = piece.symbol()
                cells.append(UNICODE_PIECES[sym] if use_unicode else sym)
        lines.append(f"{rank + 1} " + " ".join(cells))
    lines.append("  a b c d e f g h")
    return "\n".join(lines)


def print_outcome(board):
    print("\n" + "=" * 42)
    if board.is_checkmate():
        winner = "White" if board.turn == chess.BLACK else "Black"
        print(f"Checkmate! {winner} wins.")
    elif board.is_stalemate():
        print("Stalemate - draw.")
    elif board.is_insufficient_material():
        print("Draw - insufficient material.")
    elif board.is_seventyfive_moves():
        print("Draw - seventy-five-move rule.")
    elif board.is_fivefold_repetition():
        print("Draw - fivefold repetition.")
    elif board.can_claim_threefold_repetition():
        print("Draw - threefold repetition.")
    elif board.can_claim_fifty_moves():
        print("Draw - fifty-move rule.")
    else:
        print("Game over.")
    print(f"Result: {board.result(claim_draw=True)}")
    print("=" * 42)


# --------------------------------------------------------------------------- #
# Game loop
# --------------------------------------------------------------------------- #
def play(cfg, pgn_path):
    client = LLMClient(cfg)
    board = chess.Board()

    game = chess.pgn.Game()
    label = f"{cfg['provider']}:{cfg['model']}"
    game.headers["Event"] = "CHessai self-play"
    game.headers["White"] = label
    game.headers["Black"] = label
    node = game

    print(f"\nCHessai - {label} plays itself")
    print(f"(temperature={cfg['temperature']}, max_retries={cfg['max_retries']}, "
          f"fallback={'on' if cfg['fallback_random'] else 'off'})\n")
    print(render_board(board, cfg))

    interrupted = False
    try:
        while not board.is_game_over(claim_draw=True) and board.ply() < cfg["max_moves"]:
            color = "White" if board.turn == chess.WHITE else "Black"
            print(f"\nMove {board.fullmove_number} - {color} to play...")
            move, comment, attempts = get_ai_move(client, board, cfg)
            san = board.san(move)
            board.push(move)
            node = node.add_variation(move)
            note = f"  (after {attempts} tries)" if attempts > 1 else ""
            print(f"  {color} plays {san}  [{move.uci()}]{note}")
            if comment:
                print(f'    "{comment}"')
            print(render_board(board, cfg))
            time.sleep(cfg["move_delay"])
    except KeyboardInterrupt:
        interrupted = True
        print("\n\nInterrupted - saving the game so far...")
    except ConnectionError as exc:
        print(f"\n\nLost contact with the model: {exc}")
        print(f"Is it running?  Try:  ollama serve   and   ollama pull {cfg['model']}")
        game.headers["Result"] = "*"
        save_pgn(game, pgn_path)
        return

    if board.is_game_over(claim_draw=True):
        print_outcome(board)
        game.headers["Result"] = board.result(claim_draw=True)
    elif interrupted:
        game.headers["Result"] = "*"
    else:
        print(f"\nStopped at the {cfg['max_moves']}-ply safety cap.")
        game.headers["Result"] = "*"

    save_pgn(game, pgn_path)


def save_pgn(game, pgn_path):
    if not pgn_path:
        pgn_path = f"game_{time.strftime('%Y%m%d_%H%M%S')}.pgn"
    try:
        with open(pgn_path, "w", encoding="utf-8") as fh:
            print(game, file=fh)
        print(f"PGN saved to {pgn_path}")
    except OSError as exc:
        print(f"Could not save PGN: {exc}")


# --------------------------------------------------------------------------- #
# Setup / CLI
# --------------------------------------------------------------------------- #
def load_config(args):
    cfg = dict(DEFAULT_CONFIG)
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, "r", encoding="utf-8") as fh:
                user_cfg = json.load(fh)
            cfg.update({k: v for k, v in user_cfg.items() if k in cfg})
        except (json.JSONDecodeError, OSError) as exc:
            print(f"Warning: ignoring {args.config}: {exc}")

    if args.provider:                 cfg["provider"] = args.provider
    if args.model:                    cfg["model"] = args.model
    if args.ollama_host:              cfg["ollama_host"] = args.ollama_host
    if args.base_url:                 cfg["base_url"] = args.base_url
    if args.api_key:                  cfg["api_key"] = args.api_key
    if args.temperature is not None:  cfg["temperature"] = args.temperature
    if args.max_retries is not None:  cfg["max_retries"] = args.max_retries
    if args.move_delay is not None:   cfg["move_delay"] = args.move_delay
    if args.max_moves is not None:    cfg["max_moves"] = args.max_moves
    if args.no_fallback:              cfg["fallback_random"] = False
    if args.unicode:                  cfg["unicode_board"] = True
    if args.ascii:                    cfg["unicode_board"] = False

    if not cfg.get("api_key"):
        cfg["api_key"] = os.environ.get("OPENAI_API_KEY", "") or \
                         os.environ.get("CHESSAI_API_KEY", "")
    return cfg


def preflight(cfg):
    """Fail early with a helpful message if the backend isn't reachable."""
    if cfg["provider"] == "ollama":
        tags_url = cfg["ollama_host"].rstrip("/") + "/api/tags"
        try:
            resp = requests.get(tags_url, timeout=5)
            resp.raise_for_status()
        except requests.RequestException:
            print(f"ERROR: cannot reach Ollama at {cfg['ollama_host']}.")
            print("  - Install it from https://ollama.com")
            print("  - Make sure it is running (the 'ollama' app or 'ollama serve')")
            print(f"  - Pull your model:  ollama pull {cfg['model']}")
            sys.exit(1)
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        if names and not any(n == cfg["model"] or n.startswith(cfg["model"] + ":")
                             for n in names):
            print(f"Warning: model '{cfg['model']}' not found in Ollama.")
            print(f"  Available: {', '.join(names) or '(none)'}")
            print(f"  Pull it with:  ollama pull {cfg['model']}\n")
    else:
        if not cfg.get("api_key"):
            print("Warning: provider 'openai' selected but no API key found "
                  "(set --api-key or the OPENAI_API_KEY env var).\n")


def list_ollama_models(cfg):
    try:
        resp = requests.get(cfg["ollama_host"].rstrip("/") + "/api/tags", timeout=5)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        print("Installed Ollama models:" if names else "No Ollama models installed.")
        for name in names:
            print(f"  - {name}")
    except requests.RequestException as exc:
        print(f"Could not reach Ollama: {exc}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Local AI-vs-AI chess. One model plays both sides; "
                    "every move is legality-checked and re-prompted if illegal.")
    p.add_argument("--provider", choices=["ollama", "openai"],
                   help="backend type (default: ollama)")
    p.add_argument("--model", help="model name, e.g. mistral, llama3, gpt-4o-mini")
    p.add_argument("--ollama-host", dest="ollama_host",
                   help="Ollama base URL (default: http://localhost:11434)")
    p.add_argument("--base-url", dest="base_url",
                   help="OpenAI-compatible base URL (default: https://api.openai.com/v1)")
    p.add_argument("--api-key", dest="api_key",
                   help="API key for the 'openai' provider (or set OPENAI_API_KEY)")
    p.add_argument("--temperature", type=float)
    p.add_argument("--max-retries", dest="max_retries", type=int,
                   help="re-prompts for a legal move before random fallback")
    p.add_argument("--move-delay", dest="move_delay", type=float,
                   help="seconds to pause between moves")
    p.add_argument("--max-moves", dest="max_moves", type=int,
                   help="safety cap in plies (half-moves)")
    p.add_argument("--no-fallback", action="store_true",
                   help="error out instead of playing a random move when the AI fails")
    p.add_argument("--unicode", action="store_true", help="draw board with chess glyphs")
    p.add_argument("--ascii", action="store_true", help="draw board with letters (default)")
    p.add_argument("--config", default="config.json", help="path to JSON config")
    p.add_argument("--pgn", help="where to save the game PGN")
    p.add_argument("--list-models", action="store_true",
                   help="list installed Ollama models and exit")
    return p.parse_args(argv)


def main():
    # Make unicode pieces / output safe on Windows terminals.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    args = parse_args()
    cfg = load_config(args)

    if args.list_models:
        list_ollama_models(cfg)
        return

    preflight(cfg)
    play(cfg, args.pgn)


if __name__ == "__main__":
    main()
