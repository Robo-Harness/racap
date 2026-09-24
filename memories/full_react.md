# Full ReAct task and mechanism experience

- Copy every source, support, destination, and moving-part noun phrase from the
  instruction. Keep front/middle/back/left/right attached to the noun it
  selects. Do not transfer a source selector to the destination.
- Plan only requested actions and unavoidable physical prerequisites. Open a
  closed receptacle before transport, close it only after transport succeeds,
  and set an appliance control before cookware can occlude it.
- A child controller receives only one semantic subgoal. Once this controller
  has verified a prerequisite, do not ask the child to repeat it.
- Resolve pronouns by causal role. In “close the drawer and put the bowl on top
  of it”, “on top” is an exposed support relation, not containment in the
  drawer opening.
- Use `stack` only when the instruction explicitly says stack. For “stack A on
  B and place them in C”, first move support B into C, then stack A on B.
- Treat check-state motion as evidence, not hidden truth. The home-cleared
  image is the semantic authority, but demand a persistent cavity and
  substantial panel displacement for `open`, and a flush panel without a
  cavity for `closed`.
- State retries must change contact family. For prismatic mechanisms admissible
  contacts are side_bar, graspgen_6d, hook, graspnet_6d, pca_axis, top_down.
  For revolute mechanisms use graspgen_6d, graspnet_6d, pca_axis, or top_down.
  Appliance controls use pca_axis or top_down. Never request an inadmissible or
  already attempted contact.
- Closing a drawer is normally a compressive push on its external front face;
  opening requires tensile handle retention. Closing a hinged door is a sweep
  of its external panel around the hinge.
- If the requested state is already visibly satisfied at preflight, confirm it
  without motion. If several distinct contacts produce no visual state change,
  stop rather than repeat the same path in different words.
