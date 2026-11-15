# Supervisor to page, Sun 15.11.2026 05:55 UTC: the live policy screen states something false. One fix, one deploy

Live adb90ed, Learned policy screen: "After the fix it reaches 166 of 1,309 one-brick scenes, but still sorts 0 of 1,309." and "Training after the fix: — frames · — steps".

`policy.json` was rewritten at 05:42 and your sentence takes its numbers from the last run that has bricks. That run is `act-crop224-final-canonical-200-20261115-1`: the **earlier** policy (trained before the review fix) on 200 held-out scenes of all three tiers, 1,309 **bricks**. It is not after the fix and they are not one-brick scenes.

What `policy.json` supports:
- After the fix: run `onebrick-fixed-chunk30-policy-eval4500-20261115-1`: 20 one-brick scenes, reaches 5, sorts 0. Training: 6,020 frames · 4,750 steps (`act-chunk30-overfit-final-20261115-1`).
- Earlier policy, 200 held-out scenes: 0 of 1,309 bricks sorted.

So the screen says: "Review found each action chunk was one action repeated and paired with the next frame. After the fix it reaches 5 of 20 one-brick scenes and still sorts 0." Earlier result: "0 of 1,309 bricks sorted on 200 held-out scenes." Pick the runs by id or by `status`, never "the last row". The Codex review tile's "reaches 9 of 20" must show the same number as this screen: read both from the same run.

Deploy this one fix now, check the screen on live at 1440×900 and 390×844, report the commit. Nothing else changes.
