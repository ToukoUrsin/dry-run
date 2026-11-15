# Supervisor to page, Sun 15.11.2026 01:03 UTC: framing, top bar, tiers, phone are right now. The arm is not

Opened at 01:02 UTC (`/bulk3/runs/dryrun/review_live_0107.png`, `review_live_0107_phone.png`). Framing, shadows, tier cards, top bar and the phone layout are good. Keep them. Three things left, in this order:

1. **The arm looks like crumpled foil. This is the one thing every visitor sees.** The cause is in `scripts/export_mujoco_glb.py`: `simplify_mesh` clusters vertices on a grid down to 18,000 triangles per part, then, if that is still too many, keeps every Nth face (that tears holes), and then smooths normals across the damaged surface. The SO-101 STL files hold 760,438 triangles in total; that is not much. Do this instead:
   - No grid clustering and never drop faces by subsampling. Either export the arm at full resolution (weld identical vertices, 16-bit quantised positions; a browser draws 760k triangles easily and the file compresses to a few MB with gzip or meshopt), or reduce with a real quadric-error simplifier (`pyfqmr` or `fast-simplification` from PyPI, or Open3D) to no fewer than 300k triangles for the whole arm. I withdraw the 200k cap from prompt 023.
   - Normals: split at a 30° crease angle (hard edges stay hard, curved faces stay smooth). Printed parts are flat-sided; flat shading is better than what is there now.
   - Materials from the model: yellow printed parts, black servos, black pads. Slight roughness, no metal.
   - Bowls: a smooth ring is fine for display (same radius and wall height as the 24 collision boxes). Say on the Method page that bowl walls are drawn smooth and collide as 24 boxes.
   - Compare your viewer with `/bulk3/runs/dryrun/insp_present_reference.png` (MuJoCo's own render of the same arm). The arm in the browser must look as clean as that.
2. **The right-hand "382 / 382 · commit · benchmark by tier" block repeats the tier cards.** Remove it; put the commit under the cards in small type. Let the tier cards run the full width.
3. **Inspection panel before the first `present`:** it shows a dark frame with a yellow sliver. Until the first `present` event show the panel empty with one line ("The inspector looks when the arm holds a brick up"), then the present frame with verdict and score as in prompt 025 item 4.

Main is about to commit new touching and pile numbers from a better expert (prompt 027) with the old ones kept as "before". When `/api/benchmark-tiers` carries both, show before → after on the touching and pile cards, and offer a touching replay and a pile replay next to the loose one (the arm pushes clusters apart; that is the thing to watch).

Proof as before: both screenshots, after deploy, in `/bulk3/runs/dryrun/page/`. Target: live by 01:40 UTC.
