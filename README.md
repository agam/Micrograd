
## Overview

This is a simple implementation of a Multilayer Perceptron (MLP) using the micrograd library, inspired by https://youtu.be/VMj-3S1tku0 by Andrej Karpathy.

## Running/Usage

```
go run ./cmd/main
```

Modify `main` to call other test methods as needed.

## Visualizing (when using `WriteDot`)

```
dot -Tsvg -O output.dot
```

Note: You will need `brew install graphviz` for this.

## Sample run

```
Iter 0 Loss => 4.688703
Iter 1 Loss => 3.604720
Iter 2 Loss => 2.980677
Iter 3 Loss => 2.276758
Iter 4 Loss => 1.352874
Iter 5 Loss => 0.874960
Iter 6 Loss => 0.283593
Iter 7 Loss => 0.178083
Iter 8 Loss => 0.136367
Iter 9 Loss => 0.109319
Iter 10 Loss => 0.090561
Iter 11 Loss => 0.076888
Iter 12 Loss => 0.066536
Iter 13 Loss => 0.058457
Iter 14 Loss => 0.051997
Iter 15 Loss => 0.046727
Iter 16 Loss => 0.042354
Iter 17 Loss => 0.038675
Iter 18 Loss => 0.035540
Iter 19 Loss => 0.032840
```
