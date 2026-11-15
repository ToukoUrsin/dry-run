# Supervisor to page, Sun 15.11.2026 02:01 UTC: redesign the page around the 3D. This replaces every layout note so far

Touko looked at the live page at 01:58 and said: "the 3D stuff looks good but the UI just makes the whole thing look super duper weak." He is right. It reads like an admin dashboard: small grey cards, tiny captions, ten nav tabs, raw floats ("471.75459324999997 ms"), an inspection panel stuck on "Loading the recorded inspection frame…", tier cards that say "before 382/382 → after 380/382", and a Messier section that is one small replay in empty space. His capture: `/bulk2/runs/junction/dryrun-check/page.png`. Open it before you start.

Touko films this page this morning. The film opens with a 10 s cold open and cuts straight to your first screen. Look at the frame it ends on: `/bulk3/runs/opens/final/dryrun/last.jpg` (a light grey SO-101 and one lime brick on glossy black). Your first screen has to hold up next to that frame. The entry's main goal is Best Use of Codex.

## The look

- **Dark, one accent.** Page background `#07080a`. Text `#f4f5f6`, secondary `#8b9096`. One accent, the lime of the brick in `last.jpg` (about `#a8e23a`); red only for failures. No card borders, no grey panels, no box shadows on boxes. Space and type size make the hierarchy.
- **Type.** One family (Inter or Geist, self-hosted or system fallback). Section numbers at `clamp(88px, 15vw, 220px)`, weight 600, tight tracking, tabular figures. Captions 15–17 px. Nothing on the page below 13 px.
- **Words.** At most one sentence of 14 words under a number. No raw ids, no snake_case, no commit hashes in body text (one small "commit abc1234" link per section footer is fine). Every number rounded for a person: `99 %`, `0.47 s`, `11,798 rows`. No value on the page may have more than one decimal. Write a test for that over the rendered page text.

## First screen: the station, full-bleed

- The 3D canvas fills the whole viewport (`100vw × 100svh`), no frame, no header bar over it. The table is a lit surface that fades into the black; soft contact shadows; a slow camera drift (a few degrees, 20 s) when nobody drags. Arm printed parts light grey like `last.jpg`, servos near black. Smooth the bowls (64+ segments).
- Main is adding a station look that matches the real Junction station (wooden table, an orange plate for defective and a teal plate for good), prompt 039. The replay file will carry the look's colours and table texture; draw the scene from them and fall back to the present colours. When that replay exists it is the first-screen replay.
- Over the canvas, nothing boxed: wordmark "Dry Run" top left; top right one text link "Built by Codex ↓". Bottom left, the line "Junction's brick-sorting station, as a simulator." and under it three numbers in a row from `tiers.json`: `99 %` loose · `91 %` touching · `67 %` pile, each with its two-word label and "bricks in the right bowl, scenes never seen". Bottom right, the two camera views as two small rounded pictures floating over the scene (top camera, inspection camera), each with a three-word label. The inspection picture always shows a frame: the latest `present` frame, and before the first one the first `present` frame of the replay at 40 % opacity with "waits for a brick". Never a loading sentence.
- The timeline is one thin line along the bottom edge with event ticks; controls appear on hover or tap. A small counter "good 2 · defective 1" by the bowls or in the corner.
- Phone (390 px): canvas full screen, numbers stacked bottom left, cameras as a two-picture strip under the first screen.
- The first screen must still draw with every API down (rule from prompt 034).

## Then one screen per section, in this order

Each section: `min-height: 100svh`, one huge number, one sentence, one visual that fills the rest. No tables unless named below. A section whose JSON is missing is not drawn.

1. **Built by Codex** (second screen, the one a judge must remember). Number: commits (live count from `codex.json`), with "5 Codex sessions · one night · every line". Visual: **the night in one picture**, full width: time axis 18:00 UTC → now; five lanes, one per session, every commit a tick; the supervisor's prompts as diamonds above the lanes (hover or tap shows the title, click opens the full prompt text in a side sheet); red marks where work was rejected ("first pass rejected", "0 of 50 after four hours", "live page broken"); on top the score lines by time (loose, touching, pile). Data comes from the analyst in `codex.json` (prompt 038: `night.commits`, `night.prompts`, `night.rejections`, `night.scores`). Serve prompt text from `/api/prompts/<file>`. Under it four short proof tiles (goal mode, parallel sessions, `codex review` catches, models by job), one line each, and the boundary sentence: "Everything outside `prompts/` was written by Codex. The two reference sorters under `prompts/` were ported by Codex."
2. **Scenes it never saw.** Three columns, loose / touching / pile: the big percentage, "451 of 494 bricks", a looping replay of that tier (click loads it into the first screen). Touching and pile get a second small line "first sorter: 44 %" and "first sorter: 8 %". Loose gets no arrow: 380 of 382 is the result, and one small line says "first sorter: 382 of 382". Do not draw an arrow that points down as if it were progress, and do not hide the row.
3. **Inspector.** Number: `0` with "defects missed, of 241. It only ever saw good bricks." Visual: the heat map pairs, large, four across, labels in words ("chipped stud · caught"). Then draw-your-own-defect, big, as the interactive moment: canvas left, verdict and heat map right, working on the live URL. Speed as one line: "0.47 s per brick on one laptop CPU thread".
4. **Cameras only**, **Learned policy**: only if their JSON has a measured number; show the number as it is, also when it is low.
5. **Nemotron.** One failed grab: the frames it saw, its sentence in large type, the scene it wrote, running. Number from the 448 px run when it lands.
6. **The real station.** Only when main delivers it (prompt 039): simulated top camera in the Junction look beside one still of the real top camera, label "the real station at Junction (a friend's arm, running its own code)". No real-arm claim.
7. **Tiger Data.** Number: telemetry rows, counting up from the live API. One line. A small live sparkline of rows by minute.
8. **What is simulated.** Three plain sentences. End of page: links to repo files `CODEX.md`, `prompts/`, `MORNING.md`.

Navigation: no tab bar. A column of small dots on the right edge (one per section, label on hover) is enough.

## Clock

- **02:40 UTC**: new first screen, dark theme, rounded numbers, sections 1–3 in the new layout, deployed, console clean, opened by you at 1440×900 and 390×844. Screenshots go to `/bulk3/runs/dryrun/page/`, not into git.
- **03:10 UTC**: all sections, the night picture, phone pass, deployed. After 03:10 only number refreshes and bug fixes; Touko films from 03:15.
- Deploy small and often. After every deploy: open the live URL, read the console, look at the first screen. A broken first screen at filming time is the worst outcome; a missing lower section is not.
