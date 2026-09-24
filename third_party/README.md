# Third-party source

RACaP uses RATs simulator adapters, perception services, and robot primitives.
Project-specific code lives in `racap/`, `policies/`, and `evolution/`.

| Component | Source | License |
| --- | --- | --- |
| RATs | [Playful-RATs/RATs](https://github.com/Playful-RATs/RATs) | [Apache-2.0](rats/LICENSE) |
| CaP-X | Source included within the RATs distribution | [MIT](rats/capx-baseline/LICENSE) |
| PyRoKi snippets | [chungmin99/pyroki](https://github.com/chungmin99/pyroki) | MIT; license included beside each copy |

The RATs source snapshot is `1df65a180562e91911214fa1caba7c7ed9407b3d`.
Upstream commit `d21122fb33f6717a2a1ec23b8c4f642540a75991` adds only the
Apache-2.0 license to the same source tree. This package includes that text.
CaP-X retains its separate copyright notice and license.

The vendored sources include RACaP changes for simulator/evaluator contracts,
timeouts, model routing, and portable configuration. Personal debug paths and
generated data are excluded. Names in third-party code identify upstream projects.
Downloaded simulator/perception repositories carry their own licenses and
asset access conditions. `scripts/bootstrap.sh` records the source pins.
Generated skill libraries and benchmark records are external assets; see
`docs/REPRODUCIBILITY.md`.
