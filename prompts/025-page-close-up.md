# Supervisor to page, Sun 15.11.2026 00:26 UTC: real robot is on the page now. Good. Now make it the thing people look at

I opened https://dryrun-junction-2026.fly.dev/ at 00:20 UTC (screenshots: `/bulk3/runs/dryrun/review_live_0020.png`, `review_live_0020_phone.png`). The replay is real, it plays, the verdict card and the counters are there, the tier numbers are honest. Keep all of that. What is still wrong, most important first:

1. **The robot is a speck.** Arm, bowls and bricks fill about a tenth of the viewer; the rest is empty grey table. The whole scene is 0.75 m wide (bowls at y ±0.27, bricks out to x 0.28). Frame it: look at (0.14, 0, 0.06), camera about 0.75 m away, 30° above the table, from the front-right, so arm + both bowls + all bricks fill at least 70 % of the viewer width at 1440 px and at 390 px. Slow automatic orbit of ±12° around that view; drag to orbit, never zoom further out than the start view. Draw only the work area of the table (about 0.9 × 0.9 m, rounded or faded edge), not the whole slab and the back wall.
2. **The arm looks crumpled.** It is a yellow low-polygon lump. In MuJoCo's own render the same model is clean: flat yellow printed parts, black servos, black pads (see `/bulk3/runs/dryrun/insp_present_reference.png`). Export the visual meshes (not collision hulls), spend the triangle budget on the arm (bricks and bowls are cheap), keep hard edges hard (split normals at about 30°), and use the model's own materials. Bricks need visible studs.
3. **Light.** One key light with soft shadows on the table, a weak sky fill, filmic tone mapping. Matte light table. The arm and bricks should cast shadows; right now everything is flat.
4. **The inspection panel shows the wrong moment.** At "Brick 2 · present" it shows a brick cut off at the bottom edge and dark shapes. The frame the inspector really gets at a `present` event has the brick filling the picture with the scribble readable (same reference image, four `present` frames of seed 1001). Show the frame captured at the latest `present` event, hold it until the next one, and put the verdict and score on it. Label pose A or B. Brighten the top view.
5. **It reads as an admin dashboard.** Drop the left sidebar ("Workspace / Tools / Record"); use a short top bar. The viewer gets the width.
6. **Phone.** The first phone screen is only a headline. Make the headline two lines at most, put the three tiers in one row, and have the viewer start inside the first screen.
7. **Tiers as the story.** Under the headline: three bars, loose / touching / pile, with count and per cent, read from `/api/benchmark-tiers` (main is about to improve touching and pile; the page must pick up new numbers without an edit). One line under them: "Scripted expert with ground-truth labels. Simulated."
8. Small: one play/pause button, not two. Say what the orange dots on the timeline are (colour them by event: pick, present, drop, fail).

Open it yourself at 1440 × 1000 and 390 × 844 after deploying and compare with this list before you say it is done. Proof: both screenshots in `/bulk3/runs/dryrun/page/`, paths in `status/M3.md`.

Target: live by 01:10 UTC. I will open it then.
