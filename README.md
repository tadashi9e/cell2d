# 2D Cellular Automata

Cellular automaton around von Neumann neighborhood. Each cell has a binary value of 0 or 1.

This program determines the next step cell state {0, 1} for each state {0, 1} of the cells adjacent to the center (C) and in the east-west-south-north (EWSN) directions, given as a 32-bit C⊗W⊗S⊗E⊗N rule (for the given rule, if the corresponding bit for the current state is 1, the next step state is set to 1; otherwise, it's set to 0).

Regarding the initial values, the peripheral parts are value 0, and the central part consists of cells with randomly chosen values from {0, 1}.

A cell value of 0 is represented as black, a cell value of 1 as white, and if the cell value changed from the previous step, it is shown in red.

## Requirement

- Python3
- Cupy
- OpenCV python
- Pillow

## Usage

```
usage: cell2d.py [-h] [--width WIDTH] [--rule RULE] [--height HEIGHT] [--loop LOOP] [--headless] [--animation]

options:
  -h, --help       show this help message and exit
  --width WIDTH    Field width
  --height HEIGHT  Field height
  --size SIZE      Magnify cell size (default 1)
  --rule RULE      Rule (default rule is random)
  --loop LOOP      loop count
  --batch          batch mode (run without graphics)
  --animation      animation
```

Rule value must be a 32-bit integer (ex. `0xbbee1d8a`). If rule not specified, random value 

Field size can be specified by WIDTH and HEIGHT.

## Experimental feature of this program

I wanted to create a program that automatically explores rules for cellular automata that exhibit interesting behavior.

Therefore, I display the entropy, sticky rate, and score at the end of executing this program.

Entropy is the entropy of the entire screen when considering a block of 3x3 cells; if there is at least one '1' in the block, the entropy is 1, otherwise, it's 0.

The sticky rate is the ratio of cells on the screen that did not change from the previous step to the total number of cells. This metric was introduced to exclude rules where all cells on the screen repeatedly change, causing flickering.

As for scoring, I intend to evolve it heuristically, but for now, I simply calculate it as the product of entropy and sticky rate.

Batch mode is useful for repeatedly performing score calculations with random rules.
