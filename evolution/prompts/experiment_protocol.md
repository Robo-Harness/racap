# Experiment protocol

The current curriculum stage prioritizes a capability and evaluation cohort; it
does not prohibit abstractions normally associated with later stages. Follow the
evidence and change any candidate-owned layer needed for a general solution.

Before writing code, perform this reasoning internally and encode its result in
the experiment-card fields of your JSON response.

1. **Pair outcomes.** Read exact champion successes and failures plus the most
   recent candidate delta. Treat a new win and a regression as separate causal
   observations even if aggregate accuracy is unchanged.
2. **Cluster by mechanism.** Group failures by earliest phase and physical
   mechanism, not by task ID or object name. Keep genuinely different causes
   separate.
3. **Choose one leverage point.** Select the cluster with the strongest visual
   evidence, recurrence, and reachable owning layer. State the evidence keys and
   frames. Retrieved experience may suggest a hypothesis but cannot replace
   current evidence.
   The cluster condition must remain meaningful after object names are replaced
   by unseen aliases: express its physical reason through observed geometry,
   relation type, contact state, uncertainty, or a visual-agent decision.
   Object-name examples and priors may live in ReAct experience memory, where
   vision can confirm and override them; they should not become hidden Policy
   API whitelists/exclusions.
4. **Trace the interface.** Follow the runtime call from agent decision through
   API arguments, grounding, motion, report and next observation. Decide whether
   the missing degree of freedom belongs in the API interface or in agent logic.
5. **Form a falsifiable mechanism.** Use the form: WHEN condition C occurs,
   behavior B causes physical outcome F; changing B to B' should alter observable
   O. Name what result would disprove this.
6. **Design the smallest sufficient implementation.** Small means conceptually
   focused, not line-limited. Add all code needed for a complete mechanism and
   relevant tests. Avoid unrelated cleanup in the same experiment.
7. **Predict paired deltas.** List expected new wins and plausible regressions.
   Preserve parameter ranges used by champion successes unless evidence demands
   a contract change.
   Also name one held-out perturbation (unseen instance/alias, changed pose, or
   changed seed) that should preserve the physical mechanism; this can be
   recorded as a falsification test even when the current stage runner cannot
   execute it yet. A memory prior may legitimately mention a known object name,
   but its visual applicability test and agent override path must remain clear.
8. **Expose evidence.** Return phase reports and strategy choices so the next
   rollout can distinguish a bad hypothesis from a bad execution.
   Use reached EEF poses, gripper state and path-completion reports when they
   disambiguate those cases; do not invent unavailable backend calls.
9. **Update durable memory carefully.** Record capability boundaries and recovery
   principles, never benchmark answers. A one-off critic guess should remain a
   hypothesis; repeated evidence and helped/hurt counts determine trust.

Required experiment-card fields:

- `target_failure_cluster`: mechanism-level cluster, not task IDs alone.
- `evidence`: episode keys plus visible/trace observations.
- `mechanism`: causal explanation connecting code to physical behavior.
- `expected_wins`: exact paired episodes or task families predicted to improve.
- `regression_risks`: existing capabilities most exposed by the change.
- `falsification_test`: runtime observation/outcome that would refute the change.
