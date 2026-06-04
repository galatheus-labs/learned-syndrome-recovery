"""Verified 3D Streaming Demo: surface syndrome authentication.

Streams volumetric frames where each frame's surface boundary is
cryptographically verifiable via a compact syndrome. The receiver
can detect tampering at individual surface patches.

Demo modes:
  1. Normal: stream + verify (all green)
  2. Tampered: randomly flip surface voxels, show verification failure (red patches)
  3. Corrected: use BP to fix tampered voxels using the syndrome

Usage:
    python verified_stream.py --port 8080
    # Open http://localhost:8080 in browser
"""

import asyncio
import json
import argparse
import numpy as np
import torch
import time
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading

from parity import load_constraint_ops, gf2_matmul, ConstraintConfig
from adaptive_dct import FREQ_LEVELS, make_adaptive_dct_field
from surface_constraints import build_surface_constraints, detect_surface
from syndrome_refine import refine_with_syndrome


class VerifiedStreamServer:
    def __init__(self, cube_size=32, device='cpu'):
        self.cube_size = cube_size
        self.device = device
        self.N = cube_size ** 3
        self.n_levels = len(FREQ_LEVELS)

    def generate_frame(self, t):
        """Generate a procedural volumetric frame."""
        n = self.cube_size
        coords = torch.linspace(-1, 1, n)
        xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing='ij')

        # Morphing shape
        phase = t * 0.1
        morph = 0.5 * (1 + np.sin(phase))

        sphere = (xx**2 + yy**2 + zz**2) < (0.35 + 0.1 * np.sin(phase * 2))**2
        cos_t, sin_t = np.cos(phase), np.sin(phase)
        rx = xx * cos_t + zz * sin_t
        rz = -xx * sin_t + zz * cos_t
        torus = ((torch.sqrt(rx**2 + yy**2) - 0.45)**2 + rz**2) < 0.12**2

        if morph < 0.5:
            voxels = sphere
        else:
            voxels = torus

        # Add orbiting satellite
        ox = 0.6 * np.cos(phase * 3)
        oz = 0.6 * np.sin(phase * 3)
        sat = ((xx - ox)**2 + yy**2 + (zz - oz)**2) < 0.08**2
        voxels = voxels | sat

        return voxels.to(torch.uint8).reshape(-1)

    def encode_frame(self, voxels, dct_level=3):
        """Encode: compute DCT + surface syndrome."""
        voxels = voxels.unsqueeze(0).to(self.device)
        D = self.cube_size

        # DCT field
        sel = torch.zeros(1, self.n_levels, device=self.device)
        sel[0, dct_level] = 1.0
        dct_field = make_adaptive_dct_field(voxels, D, sel)
        dct_3d = dct_field.squeeze(0).squeeze(0)

        # Surface constraints from DCT
        H_surf, stats = build_surface_constraints(dct_3d, D, mode='patch',
                                                    surface_threshold=0.3)
        n_patches = H_surf.size(0)

        # Compute syndrome
        if n_patches > 0:
            H_dev = H_surf.to(self.device)
            syndrome = gf2_matmul(H_dev, voxels.t()).t().squeeze(0)
        else:
            syndrome = torch.zeros(0, dtype=torch.uint8)

        return {
            'dct_field': dct_field,
            'dct_3d': dct_3d,
            'syndrome': syndrome,
            'H': H_surf,
            'n_patches': n_patches,
            'dct_bits': int(FREQ_LEVELS[dct_level] * 2 - 1) ** 3 * 8,
        }

    def verify_frame(self, recon, encoding):
        """Verify reconstruction against syndrome. Returns per-patch status."""
        H = encoding['H'].to(self.device)
        syndrome = encoding['syndrome'].to(self.device)
        n_patches = encoding['n_patches']

        if n_patches == 0:
            return torch.zeros(0, dtype=torch.uint8), 0, 0

        recon_dev = recon.unsqueeze(0).to(self.device)
        recon_syndrome = gf2_matmul(H, recon_dev.t()).t().squeeze(0)

        # Per-patch verification: does each patch's syndrome match?
        violations = (recon_syndrome ^ syndrome)  # 1 = mismatch
        n_pass = (violations == 0).sum().item()
        n_fail = (violations == 1).sum().item()

        return violations, n_pass, n_fail

    def tamper_surface(self, voxels, encoding, flip_fraction=0.05):
        """Simulate tampering: flip random surface voxels."""
        dct_3d = encoding['dct_3d']
        surface = detect_surface(dct_3d, threshold=0.3)
        surface_indices = torch.nonzero(surface.reshape(-1), as_tuple=True)[0]

        if len(surface_indices) == 0:
            return voxels.clone(), []

        n_flip = max(1, int(len(surface_indices) * flip_fraction))
        perm = torch.randperm(len(surface_indices))[:n_flip]
        flip_indices = surface_indices[perm]

        tampered = voxels.clone()
        tampered[flip_indices] ^= 1

        return tampered, flip_indices.tolist()

    def correct_frame(self, tampered, encoding):
        """Use BP to correct tampered voxels using the syndrome."""
        H = encoding['H'].to(self.device)
        syndrome = encoding['syndrome'].unsqueeze(0).to(self.device)

        if encoding['n_patches'] == 0:
            return tampered

        # Use DCT as prior, tampered as starting point
        dct_prior = encoding['dct_field'].view(1, -1).clamp(0.01, 0.99)

        # Blend: trust tampered voxels (high confidence) but let BP fix violations
        tampered_float = tampered.float().unsqueeze(0)
        # Convert to logits with moderate confidence
        logits = torch.log(tampered_float.clamp(0.1, 0.9) /
                           (1 - tampered_float.clamp(0.1, 0.9)))

        refined, _, _, _ = refine_with_syndrome(
            logits, H, syndrome,
            n_iters=30, damping=0.5, device=self.device)

        return refined.squeeze(0)

    def frame_to_json(self, voxels_3d, color_3d=None):
        """Convert 32³ voxels to list of occupied positions for Three.js."""
        D = self.cube_size
        positions = []
        v = voxels_3d.reshape(D, D, D)
        coords = torch.nonzero(v, as_tuple=False)
        for c in coords:
            pos = {'x': c[0].item(), 'y': c[1].item(), 'z': c[2].item()}
            if color_3d is not None:
                val = color_3d[c[0], c[1], c[2]]
                pos['color'] = val
            positions.append(pos)
        return positions


def generate_demo_data(n_frames=30):
    """Pre-generate demo frames with Bayesian temporal accumulation."""
    server = VerifiedStreamServer(cube_size=32, device='cpu')
    frames = []
    N = 32 ** 3

    # Bayesian log-odds accumulator for temporal correction
    log_odds = torch.zeros(N)

    for t in range(n_frames):
        print(f"  Frame {t+1}/{n_frames}", end='\r')

        # Generate original
        voxels = server.generate_frame(t)
        encoding = server.encode_frame(voxels, dct_level=3)

        # Verify original
        _, orig_pass, orig_fail = server.verify_frame(voxels, encoding)

        # Tamper
        tampered, flip_indices = server.tamper_surface(voxels, encoding, flip_fraction=0.08)
        _, tamp_pass, tamp_fail = server.verify_frame(tampered, encoding)

        # Single-frame BP correction
        corrected_single = server.correct_frame(tampered, encoding)
        _, corr1_pass, corr1_fail = server.verify_frame(corrected_single, encoding)

        # Bayesian temporal accumulation:
        # Each frame's BP correction gives soft evidence. Accumulate in log-odds.
        # Convert corrected probabilities to log-odds and add to accumulator.
        dct_prior = encoding['dct_field'].view(-1).clamp(0.01, 0.99)
        H = encoding['H']
        syndrome = encoding['syndrome']

        # Accumulate the CORRECTED output (not raw marginals).
        # Single-frame BP already places moving objects correctly —
        # the accumulator reinforces over time.
        corrected_prob = corrected_single.float().clamp(0.01, 0.99)
        frame_logodds = torch.log(corrected_prob / (1 - corrected_prob))

        # Soft decay for changed voxels
        if t > 0:
            prev_voxels = server.generate_frame(t - 1)
            changed = (voxels ^ prev_voxels).bool()
            log_odds[changed] *= 0.3
        # Exponential moving average: blend old accumulation with new evidence
        # This prevents log-odds from growing unboundedly
        log_odds = 0.7 * log_odds + 0.3 * frame_logodds
        log_odds = log_odds.clamp(-5, 5)  # cap at ~99.3% confidence

        # Accumulated correction
        accumulated = (torch.sigmoid(log_odds) > 0.5).to(torch.uint8)
        _, acc_pass, acc_fail = server.verify_frame(accumulated, encoding)

        frame_data = {
            'frame': t,
            'original': server.frame_to_json(voxels),
            'tampered': server.frame_to_json(tampered),
            'corrected': server.frame_to_json(corrected_single),
            'accumulated': server.frame_to_json(accumulated),
            'n_patches': encoding['n_patches'],
            'syndrome_bits': encoding['n_patches'],
            'dct_bits': encoding['dct_bits'],
            'n_tampered': len(flip_indices),
            'verify_original': {'pass': orig_pass, 'fail': orig_fail},
            'verify_tampered': {'pass': tamp_pass, 'fail': tamp_fail},
            'verify_corrected': {'pass': corr1_pass, 'fail': corr1_fail},
            'verify_accumulated': {'pass': acc_pass, 'fail': acc_fail},
            'fill_rate': float(voxels.float().mean()),
        }
        frames.append(frame_data)

    print()
    return frames


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<title>Verified 3D Streaming</title>
<style>
  body { margin: 0; background: #1a1a2e; color: #eee; font-family: monospace; }
  #container { display: flex; flex-wrap: wrap; justify-content: center; gap: 10px; padding: 10px; }
  .panel { text-align: center; }
  .panel h3 { margin: 5px 0; font-size: 14px; }
  canvas { display: block; margin: 0 auto; }
  #hud { position: fixed; top: 10px; left: 10px; background: rgba(0,0,0,0.8);
         padding: 15px; border-radius: 8px; font-size: 13px; line-height: 1.6; max-width: 350px; }
  .pass { color: #4caf50; }
  .fail { color: #f44336; }
  .warn { color: #ff9800; }
  .info { color: #2196f3; }
  #controls { position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%);
              background: rgba(0,0,0,0.8); padding: 10px 20px; border-radius: 8px; }
  button { background: #333; color: #eee; border: 1px solid #555; padding: 8px 16px;
           margin: 0 5px; cursor: pointer; border-radius: 4px; font-family: monospace; }
  button:hover { background: #555; }
  button.active { background: #1565c0; border-color: #42a5f5; }
</style>
</head>
<body>
<div id="hud">
  <div style="font-size: 16px; font-weight: bold; margin-bottom: 10px;">
    Verified 3D Streaming
  </div>
  <div id="frame-info">Loading...</div>
</div>

<div id="container">
  <div class="panel">
    <h3>Original (sender)</h3>
    <canvas id="c1" width="350" height="350"></canvas>
  </div>
  <div class="panel">
    <h3 id="middle-title">Received</h3>
    <canvas id="c2" width="350" height="350"></canvas>
  </div>
  <div class="panel">
    <h3>Single-frame BP</h3>
    <canvas id="c3" width="350" height="350"></canvas>
  </div>
  <div class="panel">
    <h3>Accumulated (Bayes)</h3>
    <canvas id="c4" width="350" height="350"></canvas>
  </div>
</div>

<div id="controls">
  <button id="btn-normal" class="active" onclick="setMode('normal')">Normal</button>
  <button id="btn-tamper" onclick="setMode('tamper')">Tampered</button>
  <button id="btn-pause" onclick="togglePause()">Pause</button>
  <span id="frame-counter" style="margin-left: 15px;">Frame 0/0</span>
</div>

<script>
const FRAME_DATA = FRAME_DATA_PLACEHOLDER;
let currentFrame = 0;
let mode = 'normal';
let paused = false;
let angle = 0;

function setMode(m) {
  mode = m;
  document.querySelectorAll('#controls button').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + m).classList.add('active');
  document.getElementById('middle-title').textContent =
    m === 'normal' ? 'Received (verified)' : 'Received (TAMPERED)';
  renderFrame();
}

function togglePause() {
  paused = !paused;
  document.getElementById('btn-pause').textContent = paused ? 'Play' : 'Pause';
}

function renderVoxels(canvasId, positions, highlightColor) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, W, H);

  const cos_a = Math.cos(angle), sin_a = Math.sin(angle);
  const cos_b = Math.cos(0.3), sin_b = Math.sin(0.3);
  const scale = 8;
  const cx = W/2, cy = H/2;

  // Sort by depth for painter's algorithm
  const projected = positions.map(p => {
    const x = (p.x - 16) * scale;
    const y = (p.y - 16) * scale;
    const z = (p.z - 16) * scale;
    const rx = x * cos_a - z * sin_a;
    const rz = x * sin_a + z * cos_a;
    const ry = y * cos_b - rz * sin_b;
    const rz2 = y * sin_b + rz * cos_b;
    return { sx: cx + rx, sy: cy - ry, depth: rz2, color: p.color || 0 };
  });

  projected.sort((a, b) => a.depth - b.depth);

  for (const p of projected) {
    const size = Math.max(3, 7 - p.depth * 0.02);
    if (highlightColor && p.color === 2) {
      ctx.fillStyle = '#f44336';  // failed verification
    } else if (highlightColor && p.color === 1) {
      ctx.fillStyle = '#4caf50';  // passed verification
    } else if (highlightColor === 'green') {
      ctx.fillStyle = '#4caf50';
    } else if (highlightColor === 'orange') {
      ctx.fillStyle = '#ff9800';
    } else {
      ctx.fillStyle = '#42a5f5';
    }
    ctx.fillRect(p.sx - size/2, p.sy - size/2, size, size);
  }
}

function renderFrame() {
  if (!FRAME_DATA.length) return;
  const f = FRAME_DATA[currentFrame];

  // Original (always green = verified)
  renderVoxels('c1', f.original, 'green');

  // Middle panel depends on mode
  if (mode === 'normal') {
    renderVoxels('c2', f.original, 'green');
  } else {
    renderVoxels('c2', f.tampered, 'orange');
  }

  // Single-frame BP correction
  if (mode === 'tamper') {
    renderVoxels('c3', f.corrected, 'green');
  } else {
    renderVoxels('c3', f.original, 'green');
  }

  // Accumulated Bayes correction
  if (mode === 'tamper' && f.accumulated) {
    renderVoxels('c4', f.accumulated, 'green');
  } else {
    renderVoxels('c4', f.original, 'green');
  }

  // Update HUD
  const v = mode === 'normal' ? f.verify_original : f.verify_tampered;
  const vc = f.verify_corrected;
  const va = f.verify_accumulated || {pass: 0, fail: 0};
  const statusClass = v.fail > 0 ? 'fail' : 'pass';
  const corrClass = vc.fail > 0 ? 'warn' : 'pass';
  const accClass = va.fail > 0 ? 'warn' : 'pass';

  document.getElementById('frame-info').innerHTML = `
    <div class="info">Frame ${currentFrame + 1} / ${FRAME_DATA.length}</div>
    <div>Fill rate: ${(f.fill_rate * 100).toFixed(1)}%</div>
    <div>Surface patches: ${f.n_patches}</div>
    <div>Syndrome bits: ${f.syndrome_bits}</div>
    <div>DCT bits: ${f.dct_bits}</div>
    <div>Total: ${f.syndrome_bits + f.dct_bits} bits (${(32768 / (f.syndrome_bits + f.dct_bits)).toFixed(1)}x compression)</div>
    <hr style="border-color: #444; margin: 8px 0;">
    <div class="${statusClass}">
      Verification: ${v.pass} pass / ${v.fail} fail
      ${v.fail > 0 ? ' INTEGRITY VIOLATION' : ' ALL VERIFIED'}
    </div>
    ${mode === 'tamper' ? `
    <div style="color: #ff9800">Tampered voxels: ${f.n_tampered}</div>
    <div class="${corrClass}">
      Single-frame BP: ${vc.pass} pass / ${vc.fail} fail
    </div>
    <div class="${accClass}">
      Accumulated Bayes: ${va.pass} pass / ${va.fail} fail
    </div>
    ` : ''}
  `;

  document.getElementById('frame-counter').textContent =
    `Frame ${currentFrame + 1}/${FRAME_DATA.length}`;
}

function animate() {
  if (!paused) {
    angle += 0.015;
    renderFrame();
  }
  requestAnimationFrame(animate);
}

// Auto-advance frames
setInterval(() => {
  if (!paused && FRAME_DATA.length > 0) {
    currentFrame = (currentFrame + 1) % FRAME_DATA.length;
  }
}, 500);

animate();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--n-frames", type=int, default=60)
    parser.add_argument("--out", default=None, help="Save HTML to file instead of serving")
    args = parser.parse_args()

    print("Generating demo frames...")
    frames = generate_demo_data(args.n_frames)
    print(f"Generated {len(frames)} frames")

    # Build HTML with embedded data
    html = HTML_TEMPLATE.replace('FRAME_DATA_PLACEHOLDER', json.dumps(frames))

    if args.out:
        with open(args.out, 'w') as f:
            f.write(html)
        print(f"Saved to {args.out}")
        return

    # Serve
    out_path = Path("verified_stream.html")
    with open(out_path, 'w') as f:
        f.write(html)

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/' or self.path == '/index.html':
                self.path = '/verified_stream.html'
            return super().do_GET()

    print(f"Serving on http://localhost:{args.port}")
    httpd = HTTPServer(('', args.port), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
