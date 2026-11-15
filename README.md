# CHessai

A Python chess game where **one local AI plays both sides** and narrates the game.
By default it uses **Mistral** running on your machine through
**[Ollama](https://ollama.com)**, but you can switch models or point it at any
OpenAI-compatible API with an API key.

There are two ways to play:

- **`chess_gui.py`** - a windowed board (Tkinter) with **live AI commentary** in a
  side panel. This is the one you probably want.
- **`chess_ai.py`** - the same game in the terminal (and the shared engine).

Every move the model proposes is checked for legality with `python-chess`. If the
move is illegal or unparseable, the AI is **prompted again** (with feedback and the
list of legal moves) until it returns a legal move. After several failed attempts
it plays a random legal move so the game never gets stuck (you can disable that).

The AI also explains each move in the first person ("I'm grabbing the center..."),
which is what fills the commentary panel.

## 1. Install

```powershell
pip install -r requirements.txt
```

## 2. Get a local model (Ollama)

1. Install Ollama from https://ollama.com and make sure it's running.
2. Pull the model you want:

```powershell
ollama pull mistral
```

Check what you have installed:

```powershell
python chess_ai.py --list-models
```

## 3. Play

**Graphical (recommended):**

```powershell
python chess_gui.py
```

A window opens with the board and a commentary panel. Set the **Model / Provider**
at the bottom, then click **▶ New Game**. Use **❚❚ Pause** to pause between moves,
and the sliders to change move speed and temperature. (Tkinter ships with Python,
so there's nothing extra to install.)

**Terminal:**

```powershell
python chess_ai.py
```

The board prints after every move with the AI's commentary, and the game is saved as
a `.pgn` file at the end (open it in any chess viewer, or on
https://lichess.org/paste). Press **Ctrl+C** to stop early; the game so far is saved.

> Tip: in the GUI, the model/provider fields take effect when you press **New Game**.
> Everything else (API key, base URL, etc.) can still be set in `config.json`.

## Changing the model

Three ways, in increasing priority:

- **Edit `config.json`** (easiest): change `"model"` to e.g. `"llama3"`, `"qwen2.5"`, `"phi3"`.
- **Command line:** `python chess_ai.py --model llama3`
- A CLI flag always overrides `config.json`, which overrides the built-in defaults.

## Using an API key instead of Ollama

Point the `openai` provider at any OpenAI-compatible endpoint:

```powershell
# OpenAI
python chess_ai.py --provider openai --model gpt-4o-mini --api-key sk-...

# Or keep the key out of your shell history with an env var:
$env:OPENAI_API_KEY = "sk-..."
python chess_ai.py --provider openai --model gpt-4o-mini

# Any compatible host (OpenRouter, Together, Groq, a remote Ollama, etc.)
python chess_ai.py --provider openai --base-url https://openrouter.ai/api/v1 --model mistralai/mistral-7b-instruct --api-key ...
```

You can also set `provider`, `base_url`, and `api_key` directly in `config.json`.

## Useful options

| Flag | Meaning |
|------|---------|
| `--model NAME` | Model to use (both sides) |
| `--provider {ollama,openai}` | Backend type |
| `--api-key KEY` / `--base-url URL` | For the `openai` provider |
| `--temperature 0.3` | Lower = more deterministic play |
| `--max-retries 8` | How many times to re-ask for a legal move |
| `--no-fallback` | Stop with an error instead of playing a random move when the AI keeps failing |
| `--move-delay 0` | Speed: seconds to pause between moves |
| `--max-moves 300` | Safety cap in plies (half-moves) |
| `--unicode` / `--ascii` | Board style (letters by default; uppercase = White) |
| `--pgn game.pgn` | Where to save the game |
| `--list-models` | Show installed Ollama models |

Run `python chess_ai.py --help` for the full list.

## Notes

- LLMs are not strong chess players. The legality checker keeps the game *legal*,
  not *good* — expect plenty of blunders, especially from small local models.
- Lower `--temperature` makes the model follow the "pick a legal move" instruction
  more reliably (fewer retries), at the cost of more repetitive games.
