# Eval Set: 40 episodes (10 per task_type, 5 trials each)

Sampled from `capx_rats_all_combined_core_200_benchmark/benchmark.json` with
`random.Random(42).shuffle()`, then `[:10]` per task_type.

- task_types: ['close', 'open', 'pick', 'pick_and_place']
- episodes per type: 10
- total: 40 episodes
- per-task trials: 5 (configured via --total-trials at launch)
- total rollouts: 40 * 5 = 200

Source indices in core_200_benchmark (zero-indexed):
  close: [54, 59, 61, 66, 69, 73, 75, 76, 79, 95]
  open: [0, 1, 3, 9, 13, 15, 19, 29, 41, 42]
  pick: [100, 106, 110, 114, 117, 118, 119, 122, 123, 130]
  pick_and_place: [155, 163, 167, 172, 174, 177, 178, 187, 191, 198]
