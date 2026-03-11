# Lessons Learnt

## zheng-shang-you-oja: DouZero-Style RL Baseline

### State/Action Encoding (oja.1, oja.2)
- Card matrix encoding (15 ranks × 4 suits = 60D) works well for representing both state and actions
- Jokers need special handling: rows 13-14 with suit index 0 only
- Move history encoding needs both the card matrix AND a player ID one-hot to be useful
- State dimension for 3-player: ~1169D (dominated by the K=15 move history at 945D)

### Q-Network Architecture (oja.3)
- LSTM for history encoding is important — flattening history into MLP loses temporal structure
- The `_split_state` method that separates history from static features is critical for correct LSTM input
- Network has ~2M parameters with default settings (6-layer MLP, 512 hidden)

### Self-Play (oja.4)
- epsilon-greedy with epsilon=0.1 works better than 0.01 in early training for exploration
- Recording (state, action, reward) tuples during play is clean — just save agent's internal buffers

### DMC Training (oja.5)
- Every-visit MC with terminal reward +1/-1 and gamma=1 is conceptually simple and effective
- Training converges to beat heuristic baseline within ~100 iterations (3200 episodes)
- Loss starts ~0.6 and stabilizes around 0.5
- Mini-batch training over collected transitions works well (batch_size=256)

### Evaluation (oja.6)
- Evaluation with RL agent is much slower than random/heuristic due to network inference
- Seat rotation is important for fair evaluation in multi-player games
- 100 eval games is sufficient to see trends; 1000+ for statistical significance

### Heuristic Agent (oja.7)
- Simple strategy: lead with large low-rank combos, follow with minimum rank, conserve bombs
- Beats random agent ~40-45% in 3-player (vs 33% baseline)
- Useful as a training sanity check — RL should surpass it quickly

### Training Results
- 200 iterations, 6400 episodes, 364K transitions in ~7.5 minutes on CPU
- RL agent achieves 68% win rate vs Heuristic (29%) and Random (3%) at iter 200
- Self-play with a single shared network is sufficient for the baseline
