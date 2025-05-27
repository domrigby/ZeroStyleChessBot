# ZERO STYLE CHESS ENGINE
#####  By Dom Rigby

## Overview
This repository contains code for a Deep Reinforcement Learning agent to learn to play chess using neural network assisted 
Monte-Carlo Tree Search (MCTS), inspired by Google Deepmind's [AlphaGoZero](https://www.google.com/search?q=alpha+go+zero+paper&client=ubuntu-sn&hs=JWS&sca_esv=c9803a166ab8e7a0&channel=fs&sxsrf=ADLYWIJkjbSEES--hpoPf2U7wdxifl9_gg%3A1736708404696&ei=NBGEZ8eWKoe7hbIPx7_v4Ak&ved=0ahUKEwjHgKn87vCKAxWHXUEAHcffG5wQ4dUDCBA&uact=5&oq=alpha+go+zero+paper&gs_lp=Egxnd3Mtd2l6LXNlcnAiE2FscGhhIGdvIHplcm8gcGFwZXIyBxAjGLECGCcyCxAAGIAEGJECGIoFMgsQABiABBiGAxiKBTILEAAYgAQYhgMYigUyBRAAGO8FMggQABiABBiiBEjUCVDfBljSCHABeACQAQCYAaMBoAHUA6oBAzEuM7gBA8gBAPgBAZgCAqAChwHCAgoQABiwAxjWBBhHmAMAiAYBkAYIkgcDMS4xoAfTFg&sclient=gws-wiz-serp). This was a very engaging challenge to 
implement this on extremely hardware" inference is on my MSI laptop with a RTX 1050 and training on a rented 3090. I therefore
had to add a lot of biases to try and point it in the right direction 

I created this engine with the aim of really honing I created this engine with the aim of really honing the following skills:
1. **Parallelisation** in both PyTorch and standard Python
2. **Writing faster PyTorch code**. This was a fun and educational exercise is trying to learn something complex on extremely 
limited hardware. Due to the lack of availability of GPUs on Google Cloud, this was all trained on my 2016 gaming laptop with a 1050 GPU.
3. **Monte-Carlo Tree Search** implementation and design.
4. **C++ with Python** skills. The engine is written in C++, a language that is not my first choice.
5. More **reinforcement learning tricks**. As I will explain later, I have tried to implement some interesting things I have seen in the literature. E.g. model distillation and curiculum learning.

## To Run:
### Building C++ Engine
This library relies on C++ chess engine which I wrote (with a lot of help from the internet) to decreases chess related 
compute time. This library has to be built into an importable python library. To do this run:

Linux:
```bash cpp_chess_env/build_engine.sh```

If this fails, try running ```bash cpp_chess_env/install_cmake.sh``` to ensure cmake is installed

## Play
Run ```human_vs_machine.py```

## Training
Training followed the following steps:

1. **Behavioral Cloning**
   * Due to computing limitations, I opted to hot start the model using **imitation learning**. This involved three major steps:
     1. **Data collection**: the datasource used was the Lichess database. This however included a lot of low quality data (timeouts, resignations etc.)
     2. **Data curation**: the database has to be curated in order to only train on games with meaningful results. One move was taken from each game to try and make the data IID.
     3. **Data selection**: it is important to try and not introduce any biases from the dataset.
     4. **Training and validation**: the model was trained on this data to predict the correct move (one hot encoded) as well as the winner of the game.
        Validation was performed on a mixture of Grandmaster games and Lichess puzzles.
2. **[PENDING] Model distillation**
   * The model I trained on the rented 3090 was prohibitively slow on my puny RTX 1050. I therefore decided to try and distill this model
   model down to a lighter network.
3. **Fine-tuning via reinforcement learning**:
   * The policy you get from purely training on chess games is *pretty good*. It plays at around 1100-1300 ELO (rough guess).
   * In order to break through this ceiling we were going to need to interact with the environment and use self-play reinforcement
    learning. We do not however want to forget everything we learned in pre-training and I therefore opted to go along the more 
    **fine-tuning route**.
   * We also want to make sure we can walk before we can run. I therefore opted to add an explicit **curriculum** to the RL training. This
   was a set of increasing difficultly chess puzzles from the Lichess chess puzzles database.
   * Steps:
     1. **Self-play** using Monte-Carlo Tree Search (MCTS) and collect trajectories from these games to tune on.
     2. Opponent: starts out playing against itself. As training progresses, it chooses from a set of 10 previous past networks. I would love
     to do a proper league and then a Nash Distribution, but I do not have the spare compute!
     2. Feature extractor is originally frozen but then thawed during training to avoid catastrophic forgetting
     3. During buffer sampling, a small number of real human moves are added to the batch. The original policy from the 
     behavioral cloning is calculated and the cross-entropy error with the new policy is added as factor to the loss
     4. High regularisation and clipping is used.




