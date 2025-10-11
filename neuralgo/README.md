# NeuralGo

Tiny self-play Go laboratory inspired by sgd.fyi. A single neural value network evaluates both colours on a 9×9 board, learns via
gradient descent, and plays endless games against itself. The UI visualises win-probability traces, per-game learning trajectory,
and exposes every weight/bias parameter so you can inspect the evolving program.
