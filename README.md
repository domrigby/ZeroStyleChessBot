# ZERO STYLE CHESS ENGINE
#####  By Dom Rigby

## Overview
This repository contains code for a Deep Reinforcement Learning agent to learn to play chess using neural assisted 
Monte-Carlo Tree Search (MCTS), inspired by Google Deepmind's [AlphaGoZero](https://www.google.com/search?q=alpha+go+zero+paper&client=ubuntu-sn&hs=JWS&sca_esv=c9803a166ab8e7a0&channel=fs&sxsrf=ADLYWIJkjbSEES--hpoPf2U7wdxifl9_gg%3A1736708404696&ei=NBGEZ8eWKoe7hbIPx7_v4Ak&ved=0ahUKEwjHgKn87vCKAxWHXUEAHcffG5wQ4dUDCBA&uact=5&oq=alpha+go+zero+paper&gs_lp=Egxnd3Mtd2l6LXNlcnAiE2FscGhhIGdvIHplcm8gcGFwZXIyBxAjGLECGCcyCxAAGIAEGJECGIoFMgsQABiABBiGAxiKBTILEAAYgAQYhgMYigUyBRAAGO8FMggQABiABBiiBEjUCVDfBljSCHABeACQAQCYAaMBoAHUA6oBAzEuM7gBA8gBAPgBAZgCAqAChwHCAgoQABiwAxjWBBhHmAMAiAYBkAYIkgcDMS4xoAfTFg&sclient=gws-wiz-serp).

I created this engine with the aim of really honing my rusty Python parallelisation skills. This library will now run
training, inference and tree search in parallel, so I am happy to say I have at least got part of the way there!

**This README is not complete**

## To Run:
### Building C++ Engine
This library relies on C++ chess engine which I wrote (with a lot of help from the internet) to decreases chess related 
compute time. This library has to be built into an importable python library. To do this run:

Linux:
```bash cpp_chess_env/build_engine.sh```

## Training
There are two training modes:

1. ```train.py``` runs a single agent playing chess
2. ```parallel_train.py``` runs multiple agents playing chess and learning using the same network.

## Inference
Playing against the machine is still a pending change. It can perform self-play using ```chess_bot.py```

## Write Up Pending
I am planning to do a write-up on this soon. It will include what I learnt parallelisation and PyTorch 
optimisation.

I also intend to add a PyGame based play mode such that anyone can take it on.