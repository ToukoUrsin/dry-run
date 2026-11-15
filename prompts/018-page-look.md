# Supervisor to the page session, Sat 14.11.2026 20:42 UTC: what the live page looks like now

Full-page shot after 20 s of loading: `/tmp/drshots/pw-top.png`, `/tmp/drshots/pw-mid.png` (commit 46711bd). The data is real and connected now. Thank you for the smoke gate. (`scripts/smoke_public.py` imports `playwright`, which is not in the repo's environment: add it to `pyproject.toml` so anyone can run it.)

It is still a long report, and the 3D view is the weakest part of the first screen. In order:

1. **The 3D viewer is tilted and reads as a toy.** The horizon is rolled about 30 degrees and the table runs off the corner. MuJoCo is Z-up: set `camera.up = (0, 0, 1)`, start from a three-quarter front view about 0.6 m away aimed at the middle of the work area, orbit controls with limits so the camera cannot go under the table.
2. **The arm is one flat yellow.** Use the colours from the MJCF materials (printed parts and the black servo bodies are different materials), flat-shaded normals off, and give the meshes soft shadows with a shadow map large enough that edges are not blocky. Add a light floor grid and a soft ambient term. Look at how Soren lit theirs (ambient occlusion, antialiasing) and match that level.
3. **The bowls look like gears.** They are drawn from the 16 collision boxes. Draw each bowl as a smooth turned shape (a lathe profile) in its colour with "good" and "defective" labels; keep the collision boxes for physics only. Same idea for bricks: rounded edges, round studs.
4. **Camera panels** play at their own pace. Tie both videos to the scrubber so the three views show the same instant, and overlay the inspector heat map at `present`.
5. **The first screen** should be the viewer at most of the width with the score and the two cameras beside it, visible without scrolling at 1440×900. The headline can be one line.
6. **It is one scroll, not a workspace.** `prompts/013` asks for views with a sidebar or tabs: Station, Scenes, Scene detail, Compare, Training, Inspector, Build a station, Built with Codex, What is simulated. Each is its own route with its own empty state. Move to the real frontend build now; do not keep growing one HTML file.
7. **Empty sections should not be on the first visit.** "No reach-map artifact is attached yet", "No station requests recorded", "No failed rollout selected" (while the only rollout shown failed): hide a view from the navigation until it has data, and when the only rollout is a failure, select it.
8. The inspector table has no row and column headers a stranger understands ("actual" against "decision"), and shows the rejected first model. Label it as the first attempt with its date until the new one lands.

Take the next screenshots with a 20 s wait, look at them with `gpt-6.1-terra` against Soren's screens and linear.app, and keep the rounds in `/bulk3/runs/dryrun/page-shots/`.
