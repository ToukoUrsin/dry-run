# Two-minute film plan

Touko films this on Sunday with a phone and the screen capture built into the
page. Every score or latency shown on screen is read from the corresponding
status file; if a milestone is open, the film says so.

| Time | Shot | What Touko records |
| --- | --- | --- |
| 00:00–00:12 | Cold open | Play `/bulk3/runs/opens/final/dryrun/open.mp4` from 0:00 for its first 10 s, with Touko's line “Dry Run tests the station in simulation before the arm moves.” over it; cut for the last 2 s to the phone shot of the real marked Duplo box and SO-101 station. |
| 00:12–00:27 | First screen | The dashboard's MuJoCo station, top camera and inspection camera. Point to the current checkpoint and its commit. |
| 00:27–00:42 | One rollout | Play a replay from start to finish. Pause on approach, grasp, inspection, verdict and bowl release. Include a failed rollout if that is the available evidence. |
| 00:42–00:57 | Failure evidence | Open the failure panel, show the selected frames and Nemotron's plain-sentence diagnosis. Open the generated next-scene record. |
| 00:57–01:12 | Measured comparison | Show the Nemotron-versus-random curriculum table from M5. Read the result exactly, including a zero or an open gate. |
| 01:12–01:27 | Inspector | Show a rendered brick, its heat map and the CPU timing/confusion evidence from M2. Say that the model was trained on good renders and which tests remain open. |
| 01:27–01:42 | Data trail | Show telemetry rows, events and the aggregate-backed score. Open the replay timeline and the station sentence that produced it. |
| 01:42–01:55 | Codex build | Scroll through the relevant entries in `CODEX.md`: sessions, models, checks and review. Do not claim an entry whose proof is absent. |
| 01:55–02:00 | Real-world handoff | Hold up the marked bricks and end on `MORNING.md`: the phone-photo inspector test and the real-arm checks Touko will run. |

Use a quiet room, diffuse light and one continuous screen capture. Do not stage
a success, replace a failed rollout, or add a number that is not in a status
file.
