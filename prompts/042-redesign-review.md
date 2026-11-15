# Supervisor to page, Sun 15.11.2026 02:32 UTC: review of the redesign (live `bbf30f4`). Right direction. Fix these in this order

Looked at the live page at 1440×900, 390×844 and scrolled through. The dark page and big type work. Touko films from 03:15, so order matters: 1 → 2 → 3 first, deploy, then the rest.

## 1. First screen (it is the film's first shot)

- **Old elements are still there under the new ones.** The headline sits on top of a leftover dark box ("INSPECTION · waits for a brick"), "Starting scene" and "The recorded MuJoCo inspection verdict appears…". Remove the old replay header, caption and status box completely.
- **The 10-tab nav bar is still there.** Remove it. Wordmark left, "Built by Codex ↓" right, nothing between.
- **Camera.** The arm is cut off at the top and the green bowl at the left. Frame the whole station: arm base to top of the arm, both bowls and every brick inside the middle 70 % of the screen at 1440×900, text never over a brick or the arm. Put the text block on a dark gradient at the very bottom left, or move the scene right so the left third is free.
- **Arm and bowls.** Arm printed parts light grey (`#c9ccd1`, like `/bulk3/runs/opens/final/dryrun/last.jpg`), not pale yellow. Bowls are 24 flat boxes: draw them as smooth rings (64+ segments).
- **Numbers.** `99.5 %`, `91.3 %`, `67.1 %` as the big figures, "380 of 382 loose" style line under each. Drop "commit b9d897c · benchmark by tier" from the first screen.
- **Phone.** Headline, numbers and captions lie over the arm and each other and cannot be read. On the phone: the scene takes the top 55 % of the screen, then headline, the three numbers in one row, then the two cameras, on solid background.
- The console still shows `REQFAILED /replay-video/1001/inspection-camera.mp4` four times per load. Find it and fix it.
- Main is finishing a replay in the Junction look (wooden table, orange and teal bowls). Use it as the first-screen replay the moment `tiers.json` names it.

## 2. Built by Codex (the screen the Codex judges remember)

- **The lanes are wrong: ticks are evenly spaced, not placed by time.** Place every commit by its timestamp on the 18:00 → now axis. The real picture (long silence, then bursts; main at 125 commits, page 50) is the interesting one.
- Make the picture tall (about 60 % of the screen): hour labels on the axis; each lane labelled "main · 125 commits · GPT-6 Astra" style; **the score lines drawn above the lanes** (loose, touching, pile from `night.scores`, zero included, each end labelled with its percentage); **rejections as red vertical lines with their label written next to them** ("first pass rejected", "0 of 50 after four hours", …), not only in a legend; prompts as diamonds on their own row, title on hover, full text in a side sheet on click.
- "241" needs its word: "241 commits". Move the "commit 15dd8ee" caption from between title and number to the section footer, on every section.
- The four tiles say nothing ("One objective, kept visible."). Give each a fact from `codex.json`: hours in goal mode, number of sessions and what each owned, what `codex review` caught (one real catch), which model did which job.

## 3. Sections with wrong or weak content

- **Nemotron shows "0 %". That is a missing value drawn as zero.** `accuracy.combined` is null. The number is stage accuracy from `analyst.json` (44 % of 300). Never draw null as 0, anywhere: write a helper and a test. The large quote must be Nemotron's own sentence from an example, not the simulator's label "No clear grasp." Show three examples where it was right and one where it was wrong, each: picture, what Nemotron said, what the simulator recorded.
- **Learned policy** is ten blocks of zero bars with raw run names. Replace all of it with: the line "The learned policy does not reach a brick yet: 0 of 400." and one chart, closest approach to a brick by training step (98 → 56 mm over 21,588 steps) from `policy.json`. No run ids.
- **Cameras only**: "2.1 %" with "Three controls, measured separately" tells nobody anything. Make it a three-step ladder, large: "Told where the bricks are and which are defective: 99.5 %" → "Inspector decides the bowl from the camera: 99.5 %, 1,662 decisions, 0 wrong" → "Finds the bricks from the top camera alone: 2 of 97 so far". Numbers from `inspector.json`, as they are.
- **Scenes it never saw**: touching and pile say "Replay export pending". Main published both push replays at 02:17 (`tiers.json` → `replays`). Show them looping, large; click loads them into the first screen.
- **Inspector**: captions as words ("scribble · caught", "good brick · passed"), not "Scribble · Defective · Defective". One sentence, not three repeats of "0 of 241". Draw-your-own-defect: brick and verdict side by side, large; show the heat map of the visitor's drawing.
- **Tiger Data**: the number's line box covers the sentence under it (same on every section: give the big number `line-height: 1` and a margin). Drop "Query 1,107.1 ms". Number counts up from the live API.
- **Stations**: drop the grey "Seed 16,000 · …" paragraph and "3 · Unsupported Sentence Parameter" (say "3 rejected: sentence asked for something the checker cannot build"). Six large pictures, not twelve small ones.
- Evidence pictures load only when scrolled to. For the first two sections below the first screen, load eagerly.

## 4. Then

The real-station pair (when main delivers `real_station` in `tiers.json`), end-of-page links, the right-edge dots. Screenshots of every section at both sizes into `/bulk3/runs/dryrun/page/` and look at each one before you say done. All of this deployed by **03:10 UTC**; first screen and Codex screen by **02:50**.
