# Disjoint Playtime Set (78 episodes)

Filtered from `capx_rats_all_combined_core_200_playtime/benchmark.json` by
removing any episode whose (scene_dataset, house_index) appears in
`capx_rats_eval_core_40/benchmark.json`.

- source playtime: 100 episodes
- eval40 scenes: 37
- dropped: 22 episodes (scene overlap with eval40)
- final: 78 episodes
- task_type breakdown: {'open': 14, 'close': 15, 'pick': 25, 'pick_and_place': 24}

Guarantee: no (scene_dataset, house_index) in this set appears in
capx_rats_eval_core_40. Skills learned here cannot leak to eval via
shared scenes.
