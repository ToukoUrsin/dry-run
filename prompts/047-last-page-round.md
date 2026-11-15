# Supervisor to page, Sun 15.11.2026 03:51 UTC: last round. Live `edc5eef` is good. Six fixes, deploy, then the page is frozen at 04:15 UTC

Touko films the page right after 04:15. Do these in order; deploy after item 3 and again at the end. Anything not done at 04:15 stays as it is.

1. **The touching tile shows the real station footage** (wooden table, pink gripper) as its replay picture. That is the friend's real arm, not our touching replay. Each tier tile must show its own simulated replay (`tiers.json` → `replays`: touching-1001, pile-1045). The real footage may only appear in a section labelled "The real station at Junction".
2. **Tier replay pictures are 60 px strips.** Make each a 4:3 looping video that fills its column (about 400×300), with "Watch replay" under it, not over it. The sentence above the tiles is cut off ("Touching and pile scenes now have recorded push replays:"): one full sentence or none.
3. **First screen**: remove the label "REPLAY Current-station loose replay · seed 1001 · bowls together in front · 3 bricks" (those were my words, not page text). The console still reports failed requests for `/replay-video/1001/top-view.mp4` and `inspection-camera.mp4` on load: fix or stop requesting them. `/api/health` still answers commit `bbf30f4` after the deploy of `edc5eef`: make it report the deployed commit.
4. **The real station** section, after "Scenes it never saw": the two stills from `tiers.json` → `real_station` side by side, large, labels "The real station at Junction (a friend's arm, running its own code)" and "Dry Run, simulated". One sentence: "Same table, same bowls, same camera angle. Nothing on this page ran on the real arm."
5. **Nemotron**: drop "Combined —". In Nemotron's quoted sentences replace the raw panel names (before_grasp, end_closing, end_cycle) with words ("before the grasp", "as the jaws close", "at the end").
6. **Draw your own defect**: the right half is empty until a verdict arrives. Put a short example there before the first drawing ("Draw a mark on the brick. The inspector answers in about half a second."), and after sending show verdict, score and the heat map large.

After the last deploy: open the live URL at 1440×900 and 390×844, read the console, scroll every section, and write the commit you deployed in your last line.
