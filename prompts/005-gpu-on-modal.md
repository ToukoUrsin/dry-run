# Supervisor to Codex, Sat 14.11.2026 18:38 UTC: a GPU is available on Modal

Checked just now: `modal` with hsec `JUNCTION_MODAL_TOKEN_ID` / `JUNCTION_MODAL_TOKEN_SECRET` runs GPU functions (a T4 answered in under a minute). Verda is not available: its API key in hsec is rejected.

How to call it:

```
hsec exec --only JUNCTION_MODAL_TOKEN_ID --only JUNCTION_MODAL_TOKEN_SECRET -- bash -c \
  'MODAL_TOKEN_ID=$JUNCTION_MODAL_TOKEN_ID MODAL_TOKEN_SECRET=$JUNCTION_MODAL_TOKEN_SECRET modal run <file.py>'
```

Use it for M4, once the expert really picks bricks up (demos from an expert that scores zero teach nothing, so do not train on them):

- Generate the demos on this box (CPU, in parallel), upload the LeRobot dataset to a Modal volume, train ACT there, download the checkpoint, run the benchmark here.
- One GPU at a time, T4 or L4 or A10G. Set `timeout` on every function and stop the app when the run ends. The credits are free Junction credits and they are limited: keep the whole night under about 10 GPU-hours. If Modal answers with a spending or quota error, stop, write it in `status/`, and train on CPU.
- Name the app `dryrun-policy` and the volume `dryrun-demos`. Do not touch the other apps and volumes in the workspace (`kaiku-*` belongs to another project).
- Log the training in `status/M4.md`: GPU type, minutes, loss curve, the checkpoint each score came from.

This does not change the order. The friction grasp (M1) is still first.
