# Third-party notices

The root MIT license applies to RACaP-authored code. It does not replace the
licenses or copyright notices of dependencies.

- `third_party/rats/`: RATs source under Apache-2.0, except separately licensed
  components. See `third_party/rats/LICENSE` and `third_party/README.md`.
- `third_party/rats/capx-baseline/`: CaP-X source under MIT. Its original
  copyright notice is retained in that directory's LICENSE.
- PyRoKi example snippets copied under the `pyroki_snippets` directories retain
  their [upstream MIT license](https://github.com/chungmin99/pyroki/blob/main/LICENSE),
  included alongside each copy.
- Simulator, perception, and model packages installed by the bootstrap are
  external dependencies. Consult their distributed license and asset terms.

RACaP modifies the vendored integration for runtime/evaluation contracts,
execution budgets, configurable paths, and hosted-model routing. The optional
VAPI route requires an explicitly configured endpoint; no commercial relay
is selected by default. Generated
benchmark/skill assets are excluded from this source package.
