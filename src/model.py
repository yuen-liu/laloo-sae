"""TopK Sparse Autoencoder model"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TopKSAE(nn.Module):
    """
    TopK Sparse Autoencoder following InterProT SAE model (Adams et. al 2025)
    Architecture: input -> hidden -> top-k sparsity -> output
    """
    def __init__(self, input_dim=30, hidden_dim=120, k=6, auxk=12, batch_size=256, dead_steps_threshold=2000):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.auxk = auxk
        self.batch_size = batch_size
        self.dead_steps_threshold = dead_steps_threshold / batch_size

        # Encoder / Decoder for weights and biases
        self.w_enc = nn.Parameter(torch.empty(input_dim, hidden_dim))
        self.w_dec = nn.Parameter(torch.empty(hidden_dim, input_dim))
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim))
        self.b_pre = nn.Parameter(torch.zeros(input_dim))

        # Initialize weights
        nn.init.kaiming_uniform_(self.w_enc, nonlinearity='relu')
        self.w_dec.data = self.w_enc.data.T.clone()
        self.w_dec.data /= self.w_dec.data.norm(dim=0)

        # Track dead neurons
        self.register_buffer("stats_last_nonzero", torch.zeros(hidden_dim, dtype=torch.long))

    def LN(self, x, eps=1e-5):
        """Layer normalization to input tensor"""
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x_norm = (x - mu) / (std + eps)
        return x_norm, mu, std

    def _tie_decoder_weights(self):
        """
        Normalizes the decoder weights to have unit norm.
        This ensures that the magnitude of features is encoded in the activations,
        not the weights.
        """
        self.w_dec.data /= self.w_dec.data.norm(dim=0, keepdim=True)

    def topK_activation(self, x, k):
        """
        Apply top-k activation to the input tensor.

        Args:
            x: (BATCH_SIZE, D_EMBED, D_MODEL) input tensor to apply top-k activation on.
            k: Number of top activations to keep.

        Returns:
            torch.Tensor: Tensor with only the top k activations preserved, and others
            set to zero.
        """
        topk = torch.topk(x, k=k, dim=-1, sorted=False)
        values = F.relu(topk.values)
        out = torch.zeros_like(x)
        out.scatter_(-1, topk.indices, values)
        return out

    def auxk_mask_fn(self):
        """
        Create a mask for dead neurons.

        Returns:
            torch.Tensor: A boolean tensor of shape (D_HIDDEN,) where True indicates
                a dead neuron.
        """
        return self.stats_last_nonzero > self.dead_steps_threshold

    def forward(self, x):
        """
        Forward pass of the Sparse Autoencoder. If there are dead neurons, compute the
        reconstruction using the AUXK auxiliary hidden dims as well.

        Args:
            x: (BATCH_SIZE, D_EMBED, D_MODEL) input tensor to the SAE.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The reconstructed activations via top K hidden dims.
                - If there are dead neurons, the auxiliary activations via top AUXK
                    hidden dims; otherwise, None.
                - The number of dead neurons.
        """

        # Layer norm + center
        x_norm, mu, std = self.LN(x)
        x_norm = x_norm - self.b_pre

        # Encode
        pre_acts = x_norm @ self.w_enc + self.b_enc

        # Top-K activations
        latents = self.topK_activation(pre_acts, self.k)

        # Update dead neurons
        dead_mask_update = (latents == 0).all(dim=0)
        self.stats_last_nonzero *= dead_mask_update.long()
        self.stats_last_nonzero += 1

        dead_mask = self.auxk_mask_fn()
        num_dead = dead_mask.sum().item()

        # Decode main latents
        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu

        # Auxiliary top-k if dead neurons exist
        if num_dead > 0:
            k_aux = min(x.shape[-1] // 2, num_dead)
            aux_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            aux_latents = self.topK_activation(aux_latents, k=k_aux)
            aux_recons = aux_latents @ self.w_dec + self.b_pre
            aux_recons = aux_recons * std + mu
        else:
            aux_recons = None

        return recons, aux_recons, num_dead

    @torch.no_grad()
    def forward_val(self, x):
        """Forward for validation (no auxk, no stats update)"""
        x_norm, mu, std = self.LN(x)
        x_norm = x_norm - self.b_pre
        pre_acts = x_norm @ self.w_enc + self.b_enc
        latents = self.topK_activation(pre_acts, self.k)
        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu
        return recons

    @torch.no_grad()
    def get_acts(self, x):
        """Return latent activations"""
        x_norm, _, _ = self.LN(x)
        x_norm = x_norm - self.b_pre
        pre_acts = x_norm @ self.w_enc + self.b_enc
        latents = self.topK_activation(pre_acts, self.k)
        return latents

    @torch.no_grad()
    def encode(self, x):
        x_norm, mu, std = self.LN(x)
        x_norm = x_norm - self.b_pre
        acts = x_norm @ self.w_enc + self.b_enc
        return acts, mu, std

    @torch.no_grad()
    def decode(self, acts, mu, std):
        latents = self.topK_activation(acts, self.k)
        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu
        return recons

    @torch.no_grad()
    def reset_usage_tracking(self):
        self.stats_last_nonzero.zero_()


def loss_fn(
    x: torch.Tensor, recons: torch.Tensor, auxk: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the loss function for the Sparse Autoencoder.

    Args:
        x: (BATCH_SIZE, D_EMBED, D_MODEL) input tensor to the SAE.
        recons: (BATCH_SIZE, D_EMBED, D_MODEL) reconstructed activations via top K
            hidden dims.
        auxk: (BATCH_SIZE, D_EMBED, D_MODEL) auxiliary activations via top AUXK
            hidden dims.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The MSE loss.
            - The auxiliary loss.
    """
    mse_scale = 1
    auxk_coeff = 1.0 / 32.0

    mse_loss = mse_scale * F.mse_loss(recons, x)
    if auxk is not None:
        auxk_loss = auxk_coeff * F.mse_loss(auxk, x - recons).nan_to_num(0)
    else:
        auxk_loss = torch.tensor(0.0, device=x.device)
    return mse_loss, auxk_loss