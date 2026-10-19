# Selected ClawMark runtime

Upstream: [evolvent-ai/ClawMark](https://github.com/evolvent-ai/ClawMark),
commit `d1b641b3171e584e69a3763c269069f32a13b574`.
Copyright (c) 2025 Evolvent AI (https://evolvent.co).

This subtree retains the upstream [CC BY-NC 4.0 notice](LICENSE) and
[license terms](https://creativecommons.org/licenses/by-nc/4.0/).
The repository-root MIT license does not relicense these files.

The following files are unchanged from that upstream revision:

- `context.py`
- `sandbox/base.py`
- `state/base.py`
- `state/email/__init__.py`, `client.py`, `manager.py`
- `state/filesystem/__init__.py`, `manager.py`

The package initializers were adapted to select filesystem/email managers.
`sandbox/local.py` replaces Docker operations with in-container subprocesses and
file copies; it does not implement host isolation. CoworkEnv's release cleanup
uses literal filesystem paths for directory creation. The parent adapter and
context builder supply Harbor stage setup, readiness checks and verification.

The retained email client header credits its adaptation from Toolathlon's
`LocalEmailManager`: https://github.com/hkust-nlp/Toolathlon. That attribution is
preserved; no additional Toolathlon checkout or task data is redistributed here.

This is a selected and modified integration, not the official ClawMark runner or
an endorsement by its authors. Original archive hashes and release hashes are in
`docs/clawmark_source_manifest.json` at the CoworkEnv repository root.
