# Security

RACaP can execute model-generated Python and launch simulator and model-service processes. Run evaluation and especially evolution in a disposable container or dedicated account with only the assets and credentials required for the experiment. Candidate Git worktrees are version isolation, not a security boundary. Do not expose unrelated files, SSH agents, cloud administration keys, or host management sockets to generated code.

## Credentials and services

- Keep credentials in environment variables or ignored `configs/local.env` with mode `0600`. Do not use real keys in examples, command-line arguments, commits, screenshots, or issue reports.
- Use HTTPS for remote model providers. Loopback services may use HTTP; do not expose their ports publicly without authentication and transport protection.
- Treat prompts, model replies, rollout traces, and service logs as private. They can contain task content or provider diagnostics. Review and redact them before sharing.
- Use provider spending limits and restricted keys. Revoke a leaked key; deleting it in a later commit does not remove it from history.
- Bootstrap fetches pinned third-party code and installs dependencies. Review the script and dependency licenses before running it. A pin helps reproducibility but is not a security guarantee.

## Source sharing

`python scripts/check_release.py --tracked` scans tracked files, not ignored local data. The full-directory mode is intended for clean exports. These are heuristic hygiene checks, not a proof that arbitrary files are safe to publish. Binary documentation assets are allowed only when their checksums match reviewed content.

Do not post vulnerabilities containing credentials or private traces in public issues. Use GitHub's private vulnerability reporting if enabled; otherwise contact the repository maintainers privately to arrange a secure report. No public contact email or response-time guarantee is specified.
