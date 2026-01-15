# Projekt2 – Brettspiel-KI (AlphaZero-Style)

Dieses Repository enthält meine Abgabe für **Projekt 2 (Brettspiel-KI)** mit drei Spielen:
- **Lass die Kirche im Dorf** (`kirche/`)
- **TicTacToe (2D)** (`tictactoe/`)
- **TicTacToe 3D** (`tictactoe_3d/`)

Die KI basiert auf dem **AlphaZero-General Ansatz** (MCTS + Neural Network) und enthält Skripte für Training, Ausführung und Tests.

---

## Setup / Installation

### Voraussetzungen
- Python 3.x

### Installation
```bash
pip install -r requirements.txt
````
## Ausführen
```bash
python play_kirche.py
````

### TicTacToe (2D) starten
```bash
python main_tictactoe.py
````

### TicTacToe Training (Quick)
````bash 
python train_tictactoe_quick.py 
````

### Tests (alle Spiele)
````bash
python test_all_games.py


### Loss-Plot (optional)
````bash 
python plot_losses.py
```
