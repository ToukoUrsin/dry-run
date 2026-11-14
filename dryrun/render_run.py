"""Render a portable SVG storyboard of one full station run."""
from pathlib import Path
from .sim import scene,expert
import html,sys
seed=int(sys.argv[1]) if len(sys.argv)>1 else 7
sc=scene(seed); r=expert(sc)
W,H=960,540
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}"><rect width="100%" height="100%" fill="#10141c"/><text x="36" y="48" fill="white" font-size="28">DRY RUN · seed {seed} · scripted expert</text>']
svg += ['<rect x="50" y="90" width="860" height="380" rx="20" fill="#26303b"/>','<circle cx="780" cy="180" r="70" fill="#237a45"/><text x="750" y="185" fill="white">GOOD</text>','<circle cx="180" cy="180" r="70" fill="#9d2938"/><text x="145" y="185" fill="white">DEFECTIVE</text>']
for i,b in enumerate(sc.bricks):
 x=300+(i%6)*75; y=180+(i//6)*85; c={"red":"#e05d5d","blue":"#4c8ddd","yellow":"#e4c74a","green":"#59a878","orange":"#e08743","white":"#dfe6ee"}[b.color]
 svg.append(f'<rect x="{x}" y="{y}" width="60" height="38" rx="5" fill="{c}"/><text x="{x+20}" y="{y+25}" fill="#111">{html.escape(b.size)}</text>')
svg += [f'<text x="36" y="515" fill="#aab5c5" font-size="18">{len(sc.bricks)} bricks · {r["cycle_time_s"]}s cycle · camera views: top + inspection</text></svg>']
Path('runs/full_run.svg').write_text(''.join(svg))
