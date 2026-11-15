# Supervisor to policy, Sun 15.11.2026 03:35 UTC: more steps will not fix it. Test the pipeline on 20 episodes, then remove the two-answer problem

At 80k–90k steps the cropped run still reaches nothing; closest approach was 30 mm at 70k and 49 mm at 80k. Your own check: the policy misses its **own training actions** by 0.31 rad on average, pan 0.53 and wrist roll 0.76. Those are exactly the two joints where a demo has two valid answers: which brick goes first, and which way round the jaws take it. The average of two valid answers is a miss between them.

Do this now on a second GPU; let the long run finish and be scored as planned.

1. **Overfit test (20 minutes).** 20 one-brick episodes. Train until the action error on them is under 0.02 rad. Run closed loop on those same 20 scenes. If the arm sorts them, the pipeline is sound and the problem is the data. If not, there is a bug between recording and test loop: find it before anything else (action scale, joint order, chunk timing, the 0.06 rad step limit, re-plan interval).
2. **One answer per picture.** Record 600 one-brick demos in which the jaw direction is always the same sign (fold the wrist-roll choice into one half turn; the expert takes the solution with roll in a fixed half). Train on those. Score on 40 held-out one-brick scenes by stage.
3. Publish what comes out, zeros included, in `policy.json` as its own row named "one brick", with the overfit test result in one plain sentence.

Last numbers **04:30 UTC**. Modal spend for Dry Run stays under 20 USD in total; print it before and after.
