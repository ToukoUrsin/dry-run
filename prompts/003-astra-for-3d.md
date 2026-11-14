# Supervisor to Codex, Sat 14.11.2026 18:12 UTC: GPT-6 Astra is now available, use it for the 3D work

`gpt-6-astra` now reaches the real model (until now the pool swapped it for 6.1 Sol). Touko wants Astra to do the spatial work, the way Soren pitched it but for real. Do not interrupt what you are doing for M1; fold this in.

Use Astra (a subagent or `codex exec -m gpt-6-astra`) for the jobs where spatial reasoning matters, and keep `gpt-6.1-sol` for the rest:

1. **Geometry.** Brick bodies with studs and correct Duplo dimensions, the bowls, the camera mounts, the defect geometry (chipped stud, missing stud). Astra writes the MJCF; a check loads it in MuJoCo, measures it and renders it; Astra looks at the render and corrects it.
2. **Stations from a sentence.** Soren only varied parameters of fixed scenes. We generate new layouts: a plain sentence ("bowls behind the arm, inspection camera on the left, a tray instead of a pile") becomes a complete, valid MJCF station. Every generated station must pass an automatic check before it counts: it loads, nothing intersects at rest, every brick start position and both bowls are within the arm's reach (test with inverse kinematics), both cameras see what they should. Keep the rejected ones and the reason; report how many of how many pass.
3. **Looking at renders.** Astra reviews frames from rollouts for things that are physically wrong (bricks sinking into the table, the gripper passing through a brick, a camera pointing at nothing).

Log each Astra use in `CODEX.md` like the other features: what it was asked, what it produced, what the check said. Show the generated stations on the page (M6), with the sentence that made each one.

Astra uses more of our weekly limit than Sol. Use it for these jobs, not for ordinary code.

One more thing: `gpt-6-luna` is rate-limited on both accounts right now ("Rate limit exceeded"). If a Luna call fails, use `gpt-6.1-sol` and move on.
