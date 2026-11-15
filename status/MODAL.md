# Modal run ledger

All new policy artifacts are under `/bulk3/runs/dryrun/`. Modal maps used the
`dryrun-demos` volume and CPU containers unless noted.

- Canonical ACT policy maps: loose 80, messy 70, pile 50; checkpoint `act-crop224-long-20261115-1/act-final.pt`; manifests under `m4/modal-map/canonical-policy/`; all 1,309 bricks scored.
- Canonical expert maps: loose 80, messy 70, pile 50; manifests under `m4/modal-map/canonical-expert/`.
- DAgger checkpoint scoring: two 20-seed `.map` runs under `m4/modal-map/dagger-policy/`.
- OpenVINO full-cycle map: 80 loose seeds under `m4/modal-map/openvino-policy/`.

The exact commands, source commits, row counts, and measured outcomes are in
`status/M4.md`; no estimated dollar figure is invented here because Modal did
not return a per-run charge in the local command output.
