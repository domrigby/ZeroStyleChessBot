# ZeroStyleChessBot

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)]  
[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)]

---

## 🚀 Overview

**ZeroStyleChessBot** is a self-play deep reinforcement learning chess engine combining a convolutional neural network with Monte-Carlo Tree Search (MCTS). Inspired by Google DeepMind’s AlphaZero, it learns entirely from scratch through:

1. **Behavioral Cloning** on grandmaster games (imitation learning)  
2. **Model Distillation** for lightweight inference  
3. **Self-Play Reinforcement Learning** with curriculum learning  

All training was done on consumer-grade hardware (RTX 1050/RTX 3090), demonstrating how to extract strong performance under compute constraints.

This was a very engaging challenge to implement on extremely limited hardware… I added biases, curriculum learning, and distillation tricks to guide training.

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

2. **Build C++ Engine**
    ```
    cd cpp_chess_env
    bash build_engine.sh         # Linux
    bash install_cmake.sh        # if cmake missing

3. Create & activate Python env
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

4. Run Training
    ```
    python train.py --config configs/selfplay.yaml

Play vs. Engine

    python human_vs_machine.py

📈 Results

* ELO Improvement: approximately 1500 ELO.

🔮 Future Work

* Full league training with Nash averaging

* Enhanced curriculum using endgame tablebases

* Distributed training on multi-GPU clusters

📄 License

This project is licensed under the MIT License. See LICENSE for details.

::contentReference[oaicite:0]{index=0}