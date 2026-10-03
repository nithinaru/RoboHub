"""Tiny RunPod GraphQL client. Prefers RUNPOD_API_KEY; falls back to the macOS Keychain."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if key:
        return key
    found = subprocess.run(
        ["security", "find-generic-password", "-s", "RUNPOD_API_KEY", "-w"],
        capture_output=True,
        text=True,
    )
    return found.stdout.strip()


def gql(query, variables=None):
    key = api_key()
    if not key:
        raise RuntimeError("RUNPOD_API_KEY is not set")
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": query, "variables": variables or {}}).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
            "User-Agent": "robohub",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


if __name__ == "__main__":
    print(json.dumps(gql(sys.argv[1])))
