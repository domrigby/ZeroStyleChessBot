# ZeroStyleChessBot

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)]  
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)]

---

## 🚀 Overview

**ZeroStyleChessBot** is a self-play deep reinforcement learning chess engine combining a convolutional neural network 
with Monte-Carlo Tree Search (MCTS). Inspired by Google DeepMind’s AlphaZero, it learns to play through:

1. **Behavioral Cloning** on mixed quality games (imitation learning)
2. **Fine-tuned through Self-Play Reinforcement Learning** with curriculum learning  

All training was done on consumer-grade hardware (RTX 1050 for development/RTX 3090 for training), demonstrating how to extract strong performance under compute constraints.

This was a very engaging challenge to implement on extremely limited hardware.

> Note: this was a fun, tinkering project and the software is not always commented or written as well as I'd like! I aim to clear this up when I have time

---

## 📚 Key Features

- **End-to-End RL Pipeline**  
  - Hot-start via Behavioral Cloning from the Lichess database  
  - Progressive fine-tuning with self-play RL  
  - Curriculum of increasing puzzle difficulty  

- **Neural Network + MCTS**  
  - ResNet-style policy/value network in PyTorch  
  - PUCT-based MCTS guided by network priors  

- **Performance Optimizations**  
  - High-performance C++ core for move generation  
  - Python bindings for seamless training scripts  
  - Parallel data collection & batched GPU inference  

- **COMING SOON: Model Distillation**  
  - Compress high-capacity models into a lightweight network  
  - Enables real-time play on an RTX 1050  

---

## 🏗 Architecture

                  ┌──────────────────────────────────────┐
                  │         PyTorch Policy/Value         │
                  │     (ResNet + Dual Head: π & V)      │
                  └──────────────▲──────────────┬────────┘
                                 │              │
                                 │              │
                             priors          value
                                 │              │
                  ┌──────────────┴──────────────▼──────────────┐
                  │            MCTS (PUCT + Rollouts)          │
                  └──────────────▲─────────────────────────────┘
                                 │
                  best move → C++ Engine
                                 │
                  ┌──────────────┴──────────────┐
                  │  Custom C++ Chess Environment │
                  │  (move generation, rules, etc)│
                  └──────────────────────────────┘

---

## 🎯 Reinforcement Learning Workflow

1. **Behavioral Cloning**  
   - **Data collection:** download and filter Lichess games  
   - **Curation:** remove timeouts/resignations, ensure IID moves  
   - **Training:** predict next move and game outcome (policy & value head)  

2. **COMING SOON: Model Distillation**  
   - Train a smaller network to mimic a high-capacity “teacher”  
   - Enables fast inference on low-power GPUs  

3. **Self-Play Fine-Tuning**  
   - **Curriculum Learning:** start on easy puzzles, ramp up difficulty  
   - **MCTS Self-Play:** collect trajectories with network-guided search  
   - **Replay Buffer:** sample mix of self-play and human moves  
   - **Loss:**  
     - Cross-entropy (policy)  
     - MSE (value)  
     - Distillation regularizer to retain pre-training
---

## 🛠 Installation & Usage

1. **Clone the repo**  
   ```
   git clone https://github.com/domrigby/ZeroStyleChessBot.git
   cd ZeroStyleChessBot

2. **Create the Python env and build the C++ engine** (requires [uv](https://docs.astral.sh/uv/) and a C++17 compiler).
   `uv sync` compiles the `chess_moves` module into `.venv` and rebuilds it when the C++ source changes.
    ```
    uv sync

3. **Run tests**
    ```
    uv run pytest

4. Run Training
    ```
    uv run python paralel_train.py

Play vs. Engine

    uv run python human_vs_machine.py

📈 Results

* ELO Improvement: approximately 1500 ELO.

🔮 Future Work

* Full league training with Nash averaging

* Focus on [high entropy moves during training](https://arxiv.org/abs/2506.01939)

* Distributed training on multi-GPU clusters

📄 License

This project is licensed under the MIT License. See LICENSE for details.

## Use of LLMs

I wrote the chess playing reinforcement learning part by hand: distributed MCTS, training, evaluation server etc.

LLMs were used to create a lot of the testing and support scripts (PyGame interface, plotting etc.) and helped **a lot** to built the C++ chess simulator.