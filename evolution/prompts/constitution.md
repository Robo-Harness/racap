# RACaP evolution constitution

1. The native task predicate is evaluator-only. Candidate code may use only observations and public runtime motion/perception methods available online.
2. Policy APIs execute bounded, short-horizon physical skills. They expose semantic and geometric controls, return structured evidence and failure modes, and never decide whether the whole language task is complete.
3. The runtime agent owns semantic state, causal ordering, tool selection, retries and stopping. Geometry and semantic validators are advisory evidence, never hidden vetoes over an explicit agent decision.
4. Pickplace, insert, stack, articulate, push and control are a suggested starting vocabulary, not a fixed ontology. Add, merge, split or deprecate APIs when rollout evidence supports it.
5. Prefer mechanisms explaining a family of failures. Never branch on task IDs, benchmark order, seeds, exact object coordinates or evaluator output.
6. Every action is bounded by episode budgets and leaves a readable trace. Expensive reflection is triggered by ambiguity or failure, not blindly on every successful primitive.
7. Visual evidence outranks assumptions. The critic is advisory and has no approval or veto role.
