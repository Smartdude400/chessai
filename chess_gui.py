#!/usr/bin/env python3
"""
CHessai - graphical AI-vs-AI chess with live commentary.

A single local AI (default: Mistral via Ollama) plays BOTH sides on a Tkinter
board, narrating each move in a side panel. Every move is legality-checked by
python-chess and re-prompted if illegal (see chess_ai.py for the engine).

Run:  python chess_gui.py
"""

import json
import os
import queue
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, scrolledtext

try:
    import chess
except ImportError:
    raise SystemExit("Missing dependency 'python-chess'. Install:  pip install chess")

import chess_ai


# --------------------------------------------------------------------------- #
# Look & feel
# --------------------------------------------------------------------------- #
LIGHT_SQ   = "#EEEED2"   # cream
DARK_SQ    = "#769656"   # green
LIGHT_HL   = "#F6F669"   # last-move highlight on a light square
DARK_HL    = "#BACA2B"   # last-move highlight on a dark square
COORD_FG   = "#5b5b5b"
PANEL_BG   = "#2b2b2b"
PANEL_FG   = "#e8e8e8"

# Filled chess glyphs (U+265A..F). We use one shape per piece type and color it
# white or dark ourselves, with a 1px outline, so pieces read clearly on both
# square colors. These are the same standard chess symbols as a piece image set.
GLYPH = {"k": "♚", "q": "♛", "r": "♜",
         "b": "♝", "n": "♞", "p": "♟"}

SQUARE = 72
MARGIN = 26


class ChessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CHessai - AI vs AI")
        self.root.configure(bg=PANEL_BG)

        self.events = queue.Queue()
        self.board = chess.Board()
        self.last_move = None

        self.gen = 0                 # game generation; stale-thread events ignored
        self.worker = None
        self.stop_event = None
        self.pause_event = None

        self.defaults = self._load_defaults()
        self._build_ui()
        self._draw_board()
        self._log("Set your model below and press ▶ New Game.\n\n", "system")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, self._poll)

    # ----------------------------------------------------------------- config
    def _load_defaults(self):
        cfg = dict(chess_ai.DEFAULT_CONFIG)
        if os.path.exists("config.json"):
            try:
                with open("config.json", encoding="utf-8") as fh:
                    cfg.update({k: v for k, v in json.load(fh).items() if k in cfg})
            except (OSError, json.JSONDecodeError):
                pass
        if not cfg.get("api_key"):
            cfg["api_key"] = os.environ.get("OPENAI_API_KEY", "")
        return cfg

    def _cfg_from_fields(self):
        cfg = dict(self.defaults)
        cfg["provider"]    = self.provider_var.get()
        cfg["model"]       = self.model_var.get().strip() or "mistral"
        cfg["api_key"]     = self.apikey_var.get().strip() or cfg.get("api_key", "")
        cfg["base_url"]    = self.baseurl_var.get().strip() or cfg["base_url"]
        cfg["temperature"] = round(float(self.temp_var.get()), 2)
        cfg["move_delay"]  = round(float(self.speed_var.get()), 2)
        return cfg

    # --------------------------------------------------------------------- UI
    def _build_ui(self):
        # Board canvas (left)
        cw = MARGIN + 8 * SQUARE
        ch = 8 * SQUARE + MARGIN
        self.canvas = tk.Canvas(self.root, width=cw, height=ch,
                                bg=PANEL_BG, highlightthickness=0)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.piece_font = tkfont.Font(family="Segoe UI Symbol", size=int(SQUARE * 0.60))
        self.coord_font = tkfont.Font(family="Segoe UI", size=10)

        # Side panel (right)
        panel = tk.Frame(self.root, bg=PANEL_BG)
        panel.grid(row=0, column=1, sticky="ns", padx=(0, 10), pady=10)

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(panel, textvariable=self.status_var, bg=PANEL_BG, fg=PANEL_FG,
                 font=("Segoe UI", 13, "bold"), anchor="w", wraplength=360,
                 justify="left").pack(fill="x", pady=(0, 6))

        tk.Label(panel, text="Commentary", bg=PANEL_BG, fg="#9fcaff",
                 font=("Segoe UI", 10, "bold"), anchor="w").pack(fill="x")
        self.log = scrolledtext.ScrolledText(panel, width=46, height=20,
                                             wrap="word", font=("Segoe UI", 10),
                                             bg="#1e1e1e", fg=PANEL_FG,
                                             relief="flat", padx=8, pady=8,
                                             state="disabled")
        self.log.pack(fill="both", expand=True, pady=(2, 8))
        italic = tkfont.Font(family="Segoe UI", size=10, slant="italic")
        self.log.tag_config("white",   foreground="#7fb3ff", font=("Segoe UI", 10, "bold"))
        self.log.tag_config("black",   foreground="#ff9d7a", font=("Segoe UI", 10, "bold"))
        self.log.tag_config("comment", foreground="#d6d6d6", font=italic, lmargin1=16, lmargin2=16)
        self.log.tag_config("system",  foreground="#8a8a8a")
        self.log.tag_config("error",   foreground="#ff6b6b", font=("Segoe UI", 10, "bold"))

        # Controls
        controls = tk.LabelFrame(panel, text="Controls", bg=PANEL_BG, fg=PANEL_FG,
                                 padx=8, pady=6)
        controls.pack(fill="x", pady=(0, 6))
        btn_row = tk.Frame(controls, bg=PANEL_BG)
        btn_row.pack(fill="x")
        self.new_btn = ttk.Button(btn_row, text="▶ New Game", command=self._new_game)
        self.new_btn.pack(side="left")
        self.pause_btn = ttk.Button(btn_row, text="❚❚ Pause",
                                    command=self._toggle_pause, state="disabled")
        self.pause_btn.pack(side="left", padx=6)

        self.speed_var = tk.DoubleVar(value=self.defaults["move_delay"])
        self._labeled_scale(controls, "Pause between moves (s)", self.speed_var, 0.0, 3.0)
        self.temp_var = tk.DoubleVar(value=self.defaults["temperature"])
        self._labeled_scale(controls, "Temperature (lower = steadier)", self.temp_var, 0.0, 1.5)

        # Engine settings
        engine = tk.LabelFrame(panel, text="Engine  (applies on New Game)",
                               bg=PANEL_BG, fg=PANEL_FG, padx=8, pady=6)
        engine.pack(fill="x")

        self.provider_var = tk.StringVar(value=self.defaults["provider"])
        self.model_var    = tk.StringVar(value=self.defaults["model"])
        self.apikey_var   = tk.StringVar(value="")
        self.baseurl_var  = tk.StringVar(value=self.defaults["base_url"])

        self._field(engine, "Provider", row=0,
                    widget=ttk.OptionMenu(engine, self.provider_var,
                                          self.provider_var.get(), "ollama", "openai"))
        self._field(engine, "Model", row=1,
                    widget=ttk.Entry(engine, textvariable=self.model_var, width=22))
        self._field(engine, "API key (openai)", row=2,
                    widget=ttk.Entry(engine, textvariable=self.apikey_var, width=22, show="•"))
        self._field(engine, "Base URL (openai)", row=3,
                    widget=ttk.Entry(engine, textvariable=self.baseurl_var, width=22))

    def _labeled_scale(self, parent, text, var, lo, hi):
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill="x", pady=(4, 0))
        tk.Label(row, text=text, bg=PANEL_BG, fg=PANEL_FG,
                 font=("Segoe UI", 9)).pack(side="left")
        val = tk.Label(row, width=4, bg=PANEL_BG, fg="#9fcaff", font=("Segoe UI", 9))
        val.pack(side="right")
        val.config(text=f"{var.get():.1f}")
        ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal",
                  command=lambda _v, lbl=val, v=var: lbl.config(text=f"{float(v.get()):.1f}")
                  ).pack(fill="x")

    def _field(self, parent, label, row, widget):
        tk.Label(parent, text=label, bg=PANEL_BG, fg=PANEL_FG,
                 font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w", pady=2)
        widget.grid(row=row, column=1, sticky="e", padx=(8, 0), pady=2)
        parent.grid_columnconfigure(1, weight=1)

    # ---------------------------------------------------------------- drawing
    def _draw_board(self):
        c = self.canvas
        c.delete("all")
        for r in range(8):                  # r=0 is the top row (rank 8)
            for f in range(8):
                rank, file = 7 - r, f
                sq = chess.square(file, rank)
                x0, y0 = MARGIN + f * SQUARE, r * SQUARE
                light = (file + rank) % 2 == 1
                color = LIGHT_SQ if light else DARK_SQ
                if self.last_move and sq in (self.last_move.from_square,
                                             self.last_move.to_square):
                    color = LIGHT_HL if light else DARK_HL
                c.create_rectangle(x0, y0, x0 + SQUARE, y0 + SQUARE,
                                   fill=color, outline=color)
                piece = self.board.piece_at(sq)
                if piece:
                    self._draw_piece(x0 + SQUARE / 2, y0 + SQUARE / 2, piece.symbol())

        # coordinates in the margins
        for f in range(8):
            c.create_text(MARGIN + f * SQUARE + SQUARE / 2, 8 * SQUARE + MARGIN / 2,
                          text="abcdefgh"[f], fill=COORD_FG, font=self.coord_font)
        for r in range(8):
            c.create_text(MARGIN / 2, r * SQUARE + SQUARE / 2,
                          text=str(8 - r), fill=COORD_FG, font=self.coord_font)

    def _draw_piece(self, cx, cy, symbol):
        glyph = GLYPH[symbol.lower()]
        cy -= 2                              # nudge for visual centering
        if symbol.isupper():                 # White
            fill, outline = "#FFFFFF", "#23262B"
        else:                                # Black
            fill, outline = "#202327", "#000000"
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx or dy:
                    self.canvas.create_text(cx + dx, cy + dy, text=glyph,
                                            font=self.piece_font, fill=outline)
        self.canvas.create_text(cx, cy, text=glyph, font=self.piece_font, fill=fill)

    # ------------------------------------------------------------------- logs
    def _log(self, text, tag=None):
        self.log.config(state="normal")
        self.log.insert("end", text, tag or ())
        self.log.see("end")
        self.log.config(state="disabled")

    # --------------------------------------------------------------- controls
    def _new_game(self):
        if self.stop_event:                  # stop any running game
            self.stop_event.set()
        self.gen += 1
        gen = self.gen
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        self.board = chess.Board()
        self.last_move = None
        self._draw_board()

        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

        cfg = self._cfg_from_fields()
        self._log(f"New game - {cfg['provider']}:{cfg['model']} plays both sides "
                  f"(temp {cfg['temperature']}).\n\n", "system")
        self.status_var.set("White to move...")
        self.pause_btn.config(text="❚❚ Pause", state="normal")

        self.worker = threading.Thread(
            target=self._run_game, args=(cfg, gen, self.stop_event, self.pause_event),
            daemon=True)
        self.worker.start()

    def _toggle_pause(self):
        if not self.pause_event:
            return
        if self.pause_event.is_set():
            self.pause_event.clear()
            self.pause_btn.config(text="❚❚ Pause")
            self.status_var.set("Resumed.")
        else:
            self.pause_event.set()
            self.pause_btn.config(text="▶ Resume")
            self.status_var.set("Paused.")

    def _on_close(self):
        if self.stop_event:
            self.stop_event.set()
        self.root.destroy()

    # ------------------------------------------------------- background worker
    def _run_game(self, cfg, gen, stop_event, pause_event):
        def emit(kind, data=None):
            self.events.put((gen, kind, data))
        try:
            client = chess_ai.LLMClient(cfg)
            board = chess.Board()
            while not board.is_game_over(claim_draw=True) and board.ply() < cfg["max_moves"]:
                while pause_event.is_set() and not stop_event.is_set():
                    time.sleep(0.1)
                if stop_event.is_set():
                    return

                color = "White" if board.turn == chess.WHITE else "Black"
                emit("thinking", {"color": color, "model": cfg["model"]})

                move, comment, attempts = chess_ai.get_ai_move(client, board, cfg)
                if stop_event.is_set():
                    return

                san = board.san(move)
                num = board.fullmove_number
                board.push(move)
                emit("move", {"fen": board.fen(), "san": san, "uci": move.uci(),
                              "color": color, "comment": comment,
                              "attempts": attempts, "num": num,
                              "check": board.is_check(),
                              "over": board.is_game_over(claim_draw=True)})

                waited = 0.0
                while waited < cfg["move_delay"] and not stop_event.is_set():
                    time.sleep(0.05)
                    waited += 0.05

            if not stop_event.is_set():
                emit("end", {"text": self._outcome_text(board)})
        except ConnectionError as exc:
            emit("error", {"text": f"Cannot reach the model: {exc}\n"
                                   f"Is Ollama running? Try:  ollama serve   then  "
                                   f"ollama pull {cfg['model']}"})
        except Exception as exc:             # noqa: BLE001 - surface anything in the UI
            emit("error", {"text": f"{type(exc).__name__}: {exc}"})

    @staticmethod
    def _outcome_text(board):
        if board.is_checkmate():
            return f"Checkmate - {'White' if board.turn == chess.BLACK else 'Black'} wins!  ({board.result()})"
        if board.is_stalemate():
            return "Stalemate - draw.  (1/2-1/2)"
        if board.is_insufficient_material():
            return "Draw - insufficient material.  (1/2-1/2)"
        if board.can_claim_threefold_repetition():
            return "Draw - threefold repetition.  (1/2-1/2)"
        if board.can_claim_fifty_moves():
            return "Draw - fifty-move rule.  (1/2-1/2)"
        return f"Game over.  ({board.result(claim_draw=True)})"

    # --------------------------------------------------------- main-thread poll
    def _poll(self):
        try:
            while True:
                gen, kind, data = self.events.get_nowait()
                if gen == self.gen:
                    self._handle(kind, data)
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    def _handle(self, kind, data):
        if kind == "thinking":
            self.status_var.set(f"{data['color']} is thinking...  ({data['model']})")

        elif kind == "move":
            self.board.set_fen(data["fen"])
            self.last_move = chess.Move.from_uci(data["uci"])
            self._draw_board()

            tag = "white" if data["color"] == "White" else "black"
            note = f"  (after {data['attempts']} tries)" if data["attempts"] > 1 else ""
            check = "  +" if data["check"] and not data["over"] else ""
            self._log(f"{data['num']}. {data['color']}: {data['san']}{check}{note}\n", tag)
            if data["comment"]:
                self._log(f"{data['comment']}\n\n", "comment")
            else:
                self._log("\n")
            self.status_var.set(f"{data['color']} played {data['san']}.")

        elif kind == "end":
            self.status_var.set(data["text"])
            self._log(f"\n{data['text']}\n", "system")
            self.pause_btn.config(state="disabled")

        elif kind == "error":
            self.status_var.set("Error - see commentary.")
            self._log(f"\n{data['text']}\n", "error")
            self.pause_btn.config(state="disabled")


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.2)
    except tk.TclError:
        pass
    ChessApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
