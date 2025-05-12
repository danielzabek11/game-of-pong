# Pong AI Challenge

A Pong game where the right paddle is controlled by a Q-learning AI. Built with Pygame.

## Requirements

*   Python 3.x
*   
    pip install requirements.txt 

## Files

*   `main.py`: Run this file to play or train.
*   `pong.py`: Game logic.
*   `agent.py`: AI agent logic.
*   `config.py`: Game settings and AI parameters.


## How to Play (Human vs. AI)
    python main.py

## How to Train the AI


*   **Training (No Visuals):**
    ```bash
    python main.py train
    ```

*   **Visualized Training:**
    ```bash
    python main.py train_visual
    ```

The AI's learned data is saved in `q_table_master_best.pkl` (used for playing) and `q_table_master.pkl`. Training progress is shown in the console.