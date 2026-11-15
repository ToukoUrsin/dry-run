# Two-minute film plan

Touko films this on Sunday with a phone and the screen capture built into the
page. Every score or latency shown on screen is read from the corresponding
status file; if a milestone is open, the film says so.

| Time | Shot | What Touko records |
| --- | --- | --- |
| 00:00–00:12 | Cold open | Play `/bulk3/runs/opens/final/dryrun/open.mp4` from 0:00 for its first 10 s, with Touko's line “Dry Run tests the station in simulation before the arm moves.” over it; cut for the last 2 s to the phone shot of the real marked Duplo box and SO-101 station. |
| 00:12–00:27 | First screen | The dashboard's MuJoCo station, top camera and inspection camera. Point to the current checkpoint and its commit. |
| 00:27–00:42 | One rollout | Play a replay from start to finish. Pause on approach, grasp, inspection, verdict and bowl release. Include a failed rollout if that is the available evidence. |
| 00:42–01:02 | Nemotron (the NVIDIA beat) | One failed grab, then the failure panel: the camera frames sent to NVIDIA Nemotron (`nemotron-3.5-super-vl`), the frame it picked as evidence, and its one-sentence diagnosis. Lower-third on screen: "NVIDIA Nemotron 3.5 Super VL · failure analyst". Then open the next scene it wrote and play it running in the simulator. Line: "When a grab fails, Nemotron looks at the camera frames, says what went wrong, and writes the next scene to test." Show it working; no accuracy or curriculum numbers on screen (M5 has them, and they are weak). |
| 01:02–01:17 | Inspector | Show a rendered brick, its heat map and the CPU timing/confusion evidence from M2. Say that the model was trained on good renders and which tests remain open. |
| 01:17–01:32 | Data trail | Show telemetry rows, events and the aggregate-backed score. Open the replay timeline and the station sentence that produced it. |
| 01:32–01:50 | Codex build | Scroll through the relevant entries in `CODEX.md`: sessions, models, checks and review. Do not claim an entry whose proof is absent. |
| 01:50–02:00 | Real-world handoff | Hold up the marked bricks and end on `MORNING.md`: the phone-photo inspector test and the real-arm checks Touko will run. |

Use a quiet room, diffuse light and one continuous screen capture. Do not stage
a success, replace a failed rollout, or add a number that is not in a status
file.
