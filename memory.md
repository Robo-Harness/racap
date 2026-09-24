# Visual manipulation experience

- Preserve every relational qualifier when identifying a source or support.
  Before acting, confirm that the marked candidate is the requested
  front/middle/back/left/right instance in the full scene; a crop that merely
  has the right appearance is insufficient.
- Keep each qualifier attached to its original noun phrase during planning.
  In “put the butter at the front in the drawer”, “at the front” selects which
  butter and “in the drawer” specifies the destination; never shorten the
  source to “butter” or rewrite the destination as “front of the drawer”.
- For visually similar products, verify category and instance as two separate
  questions: first use the enlarged crop to establish the requested object
  type, then use the full scene to establish front/middle/back/left/right. A
  geometric check whose transport identity is unestablished may name the
  wrong look-alike; treat that name as a hypothesis and compare before/after
  images independently.
- Respect causal prerequisites. Open a receptacle before inserting into it and
  close it only after insertion is visibly complete. Set an appliance control
  before cookware can occlude that control.
- Reserve `stack` for an instruction that explicitly asks to stack objects.
  Ordinary “put/place X on Y” is a `pickplace` transport, including plates,
  trays, bowls, pans, and other movable supports; do not infer an unspoken
  stacking operation merely because the destination is movable.
- Treat geometric and motion reports as useful measurements, not hidden truth.
  Resolve disagreement from the current home-cleared image, while demanding
  strong visual evidence before declaring a state satisfied.
- A drawer or door is visibly open only when there is a clear persistent gap or
  cavity and substantial panel travel. A tiny handle shift, arm occlusion, or
  perspective change is not enough.
- An object is inside a drawer, slot, rack, shelf opening, or compartment only
  when its full visible footprint clears the rim/walls and it is not bridging
  an edge. Partial overlap is a recoverable miss, never completion.
- Spatial relations need margin. If a fresh check measures a reliable lateral
  miss of roughly 0.5--5 cm or places the result at the acceptance boundary,
  perform the suggested small push and check again. Do not call done merely
  because the object is generally near the reference.
- For `on` relations with a small support, touching or visibly overlapping
  the support is weaker evidence than a reliable post-motion measurement that
  the object centre remains outside tolerance. Inspect whether the measurement
  grounded the right object and support; if it did, choose retry so the runtime
  can execute its suggested micro-push. Only dismiss that advice when the
  image shows that the measurement attached to the wrong physical referent.
- When that small-support check is reliable and identity-clear, its explicit
  outside-tolerance result constrains the visual decision: edge overlap alone
  is not contrary evidence. Request the measured micro-push instead of calling
  done; reserve visual override for a demonstrably wrong referent.
- After a released object is stably near its destination, prefer one measured
  micro-push over re-grasping it. If post-release identity or geometry is truly
  ambiguous, preserve the scene and obtain a clearer view.
- Use insertion geometry for bounded front-loaded openings such as drawers,
  slots, racks, shelf apertures, cubbies, and compartments. Ordinary top-down
  placement is for exposed support surfaces, including explicit "top of".
- Closing a drawer or hinged door normally uses the external moving panel: push
  a drawer face along its slide, or sweep a door far from its hinge. Do not
  require a force-closure grasp on an interior edge merely to close it.
- A rack/shelf noun is only an insertion target when RGB-D reveals a bounded
  opening. If both the task phrase and an independently verified visual alias
  produce no opening polygon without moving the arm, treat it as an exposed
  support and try one ordinary surface placement; never repeat identical looks.
- Once two visually identical bowls or cups overlap, neither a language
  re-grasp nor a planar push can reliably isolate the upper vessel: the former
  may pick the support and the latter moves both. Preserve the scene and report
  the miss; improve the next attempt before release.
- A retry must be materially different and supported by visible evidence. If
  every distinct contact strategy makes no progress, report a primitive or
  reachability limitation instead of repeating the same motion.
- For an irregular rigid object or a thin flush package that remained empty
  after two different analytic grasp ladders, locally track its new pose and
  try one learned 6-DoF `graspnet` grasp. Do not repeat another low/top/rim
  ladder with stale pre-contact geometry.
