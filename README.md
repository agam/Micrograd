
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
