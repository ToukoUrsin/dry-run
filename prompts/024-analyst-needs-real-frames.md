# Supervisor to analyst, Sun 15.11.2026 00:17 UTC: 32-pixel frames are not evidence

You are sending Nemotron frames of 32 × 32 pixels (and 64 before that). A brick is two or three pixels wide at that size. Whatever the model answers about such a frame is a guess, and an accuracy number built on it is worthless. NVIDIA's judges will open a saved request and look at the image.

Do it properly:

1. **Render for the analyst, not for the policy.** For each failed brick cycle, re-run that seed with the same controller (rollouts are deterministic by seed) and render at **448 px or more** at the moments that matter: just before the grasp, at the end of closing, at presentation (the inspection close-up), and at the end of the cycle. Put the four frames in one 2 × 2 sheet with a small caption on each (top view, close-up, time). One sheet, one request per failure.
2. **Ask for a judgment that can be checked.** Strict schema: failure stage (no clear grasp / missed grasp / slipped / wrong verdict / dropped outside the bowl / pushed out of reach / other), what is visible that supports it, and the scene change that would train against it. Ground truth for the stage comes from the simulator's events, which Nemotron does not get. Report accuracy per stage and the confusion table, on at least 300 failures from the new expert on messy and pile scenes (and the learned policy when it exists).
3. **Rate.** About 40 requests a minute with backoff: 300 failures take ten minutes. Do not shrink images to go faster.
4. Keep five full request and response pairs (image included) in the repo as examples, and show two of them on the page: the sheet, what Nemotron said, what really happened.
5. Throw away the 32 px and 64 px runs; do not report them.
