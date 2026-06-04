"""Inference-time gradient descent refinement.

Takes the decoder's soft output and iteratively refines it using three
differentiable losses:
  1. Parity: H @ sigma(x) should match the syndrome (BP-style tanh loss)
  2. Denoiser: trained DLSS model scores how "clean" the voxels look
  3. Embedding: cosine distance to target semantic embedding

This is essentially diffusion at decode time — the same gradient signals
used during training, now applied as an iterative optimizer on the output.

Usage:
    from refine import GradientRefiner
    refiner = GradientRefiner(parity, left_inv, denoiser, embedder, ...)
    refined = refiner.refine(decoder_logits, syndrome, target_embed)
"""

import torch
import torch.nn.functional as F

from parity import gf2_matmul, project_with_left_inverse


class GradientRefiner:
    """Iterative gradient descent refinement at inference time.

    Optimizes soft voxel logits against parity, spatial, and semantic losses,
    then hard-thresholds and projects to guarantee parity satisfaction.
    """

    def __init__(
        self,
        parity,          # (M, N) uint8 parity check matrix
        left_inv,        # (N, M) uint8 left inverse
        denoiser=None,   # VoxelDenoiser model (optional)
        embedder=None,   # VoxelEmbedder model (optional)
        cube_size=32,
        lr=0.1,
        steps=50,
        parity_weight=2.0,
        denoiser_weight=1.0,
        embed_weight=1.0,
        device='cuda',
    ):
        self.parity = parity.to(device)
        self.left_inv = left_inv.to(device)
        self.parity_t = parity.t().float().to(device)  # (N, M) for BP loss
        self.M, self.N = parity.shape
        self.cube_size = cube_size
        self.lr = lr
        self.steps = steps
        self.parity_weight = parity_weight
        self.denoiser_weight = denoiser_weight
        self.embed_weight = embed_weight
        self.device = device

        self.denoiser = denoiser
        self.embedder = embedder

        # Freeze both models — we only optimize the voxel logits
        if self.denoiser is not None:
            self.denoiser.eval()
            for p in self.denoiser.parameters():
                p.requires_grad_(False)
        if self.embedder is not None:
            self.embedder.eval()
            for p in self.embedder.parameters():
                p.requires_grad_(False)

    def _parity_loss(self, logits, syndrome):
        """BP-style differentiable parity loss.

        Measures how well sigmoid(logits) satisfies H @ x = syndrome (mod 2).
        Uses the tanh product formulation for differentiable GF(2) checks.
        """
        u = -torch.tanh(logits / 2)
        log_u = torch.log(torch.clamp(u.abs(), 1e-12, 1.0))
        sign_u = torch.sign(u)

        # Product of tanh factors per check row
        log_r_mag = log_u @ self.parity_t          # (B, M)
        neg_mask = (sign_u < 0).float()
        neg_count = neg_mask @ self.parity_t        # (B, M)
        sign_r = 1.0 - 2.0 * (neg_count % 2)
        r = sign_r * torch.exp(log_r_mag)

        parity_logits = -2.0 * torch.atanh(torch.clamp(r, -0.999999, 0.999999))
        return F.binary_cross_entropy_with_logits(parity_logits, syndrome)

    def _denoiser_loss(self, probs):
        """Denoiser score: how much does the denoiser want to change the voxels?

        A clean reconstruction should pass through the denoiser nearly unchanged.
        We measure the L2 distance between input and denoiser output in prob space.
        """
        if self.denoiser is None:
            return torch.tensor(0.0, device=self.device)

        B = probs.size(0)
        x_3d = probs.view(B, 1, self.cube_size, self.cube_size, self.cube_size)
        clean_logits = self.denoiser(x_3d)
        clean_probs = torch.sigmoid(clean_logits.view(B, -1))

        # Denoiser "energy": how much correction is needed
        # Low energy = clean reconstruction, high energy = noisy
        return F.mse_loss(probs, clean_probs)

    def _embed_loss(self, probs, target_embed):
        """Cosine distance between reconstruction embedding and target."""
        if self.embedder is None or target_embed is None:
            return torch.tensor(0.0, device=self.device)

        B = probs.size(0)
        x_3d = probs.view(B, 1, self.cube_size, self.cube_size, self.cube_size)
        recon_embed = self.embedder(x_3d)
        # Cosine distance: 1 - cos_sim
        return (1 - (recon_embed * target_embed).sum(dim=-1)).mean()

    def refine(self, init_logits, syndrome, target_embed=None):
        """Run gradient descent refinement.

        Args:
            init_logits: (B, N) decoder output logits
            syndrome: (B, M) uint8 target syndrome
            target_embed: (B, D) float target embedding (optional)

        Returns:
            refined: (B, N) uint8 refined binary voxels (parity-guaranteed)
            info: dict with loss trajectory and improvement stats
        """
        B = init_logits.size(0)
        syndrome_float = syndrome.float().to(self.device)
        syndrome_uint8 = syndrome.to(torch.uint8).to(self.device)

        # Initialize optimizable logits from decoder output
        x = init_logits.clone().detach().to(self.device).requires_grad_(True)
        optimizer = torch.optim.Adam([x], lr=self.lr)

        losses = []
        for step in range(self.steps):
            optimizer.zero_grad()

            probs = torch.sigmoid(x)

            # Combined loss
            l_parity = self._parity_loss(x, syndrome_float)
            l_denoise = self._denoiser_loss(probs)
            l_embed = self._embed_loss(probs, target_embed)

            loss = (self.parity_weight * l_parity
                    + self.denoiser_weight * l_denoise
                    + self.embed_weight * l_embed)

            loss.backward()
            optimizer.step()

            losses.append({
                'total': loss.item(),
                'parity': l_parity.item(),
                'denoiser': l_denoise.item(),
                'embed': l_embed.item(),
            })

        # Hard threshold and project to guarantee parity
        with torch.no_grad():
            refined_bits = (torch.sigmoid(x) > 0.5).to(torch.uint8)
            refined_bits, _ = project_with_left_inverse(
                refined_bits, self.parity, self.left_inv, syndrome_uint8
            )

        return refined_bits, {
            'losses': losses,
            'initial_parity': losses[0]['parity'] if losses else 0,
            'final_parity': losses[-1]['parity'] if losses else 0,
            'initial_total': losses[0]['total'] if losses else 0,
            'final_total': losses[-1]['total'] if losses else 0,
        }

    def refine_from_decoder(self, x_base, decoder_logits, syndrome,
                            target_embed=None):
        """Convenience: refine starting from the decoder's residual prediction.

        The decoder predicts residual logits (what to XOR with x_base).
        We convert to full-voxel logits for optimization, then convert back.

        Args:
            x_base: (B, N) uint8 canonical base solution
            decoder_logits: (B, N) residual logits from decoder
            syndrome: (B, M) uint8 syndrome
            target_embed: (B, D) optional embedding target

        Returns:
            refined: (B, N) uint8 refined voxels
            info: dict with refinement stats
        """
        # Convert residual logits → full voxel probabilities
        # residual prob = sigmoid(decoder_logits)
        # full voxel prob = base * (1 - residual_prob) + (1 - base) * residual_prob
        # This is the XOR in probability space
        with torch.no_grad():
            res_prob = torch.sigmoid(decoder_logits.float())
            base_f = x_base.float()
            full_prob = base_f * (1 - res_prob) + (1 - base_f) * res_prob
            # Convert back to logits
            eps = 1e-6
            full_logits = torch.log(
                torch.clamp(full_prob, eps, 1 - eps)
                / torch.clamp(1 - full_prob, eps, 1 - eps)
            )

        return self.refine(full_logits, syndrome, target_embed)
