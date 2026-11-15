# Supervisor to main, Sun 15.11.2026 02:19 UTC: review of the Junction look (`film/junction-replay-debug`, `real_vs_sim.mp4` from 02:12)

I looked at the frames. Not usable yet. Compare your render with `reference/real_top.png` side by side before you answer.

1. **The wood is wrong.** Yours is saturated orange with hard plank lines. The real table is pale, grey-beige, weathered plywood: low saturation, about `#a39683` on average (the webcam reads it even greyer, median RGB 122,127,125 in the open areas), soft large stains, no regular plank lines. Make the texture match that, and check by measuring the median colour of your top-camera render against the real frame.
2. **The camera posts and the back wall took the wood texture.** They stay dark grey. Only the table top changes.
3. **The orange bowl renders as pale yellow, the teal as neon.** The light is blowing them out. In the top-camera render the bowl floor should measure close to the real plates: orange about RGB 220,105,50, teal about RGB 0,175,185. Lower the light or the specular until the measured colours are within 25 per channel.
4. **`real_vs_sim.mp4` (02:12) shows the grey lab look, tiny, with black bars.** Rebuild it: both halves the same size (960×720 each, centred on a `#07080a` canvas), the simulated top view cropped and zoomed to the work area so the arm and bricks are as large in frame as in the real picture, arm entering from the top. Rendered at 960×720 or larger, not upscaled from 320×240. Labels in the page's type, small, under each half: "The real station at Junction" and "Dry Run, simulated".
5. Pick the 12 s of the simulated replay where a brick is gripped and carried, to sit next to the real grab.

Look at the result yourself (frames at 2 s, 6 s and 10 s) before you call it done. The junction-look replay for the page's first screen needs the same fixes; publish its pointer in `tiers.json` as soon as it passes your own look. By **02:40 UTC**.
