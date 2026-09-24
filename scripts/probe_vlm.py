#!/usr/bin/env python3
"""Fail fast when the visual provider cannot return one valid JSON reply."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    from racap.backends.vlm import probe_grounder

    print(json.dumps(probe_grounder(args.model), ensure_ascii=False))


if __name__ == "__main__":
    main()
