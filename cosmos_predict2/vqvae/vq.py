import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from imaginaire.utils import log
import math


class SimVQ(nn.Module):
    def __init__(self, codebook_size: int = 8192, embedding_dim: int = 128, beta: float = 0.25):
        super().__init__()
        self.beta = beta
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim**-0.5)
        for p in self.embedding.parameters():
            p.requires_grad = False

        self.embedding_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, z: torch.Tensor):
        assert z.shape[-1] == self.embedding_dim

        z_flattened = z.view(-1, self.embedding_dim)
        quant_codebook = self.embedding_proj(self.embedding.weight)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(quant_codebook**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, einops.rearrange(quant_codebook, "n d -> d n"))
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)

        codebook_counts = torch.bincount(min_encoding_indices, minlength=self.codebook_size)
        e_mean = codebook_counts.float() / min_encoding_indices.numel()
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-8)))

        commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()

        return z_q, commit_loss, perplexity


class SimVQ_Usage(nn.Module):
    """
    SimVQ module with Exponential Moving Average (EMA) tracking for codebook usage.
    """

    def __init__(self, codebook_size: int = 8192, embedding_dim: int = 128, beta: float = 0.25, ema_decay: float = 0.99):
        """
        Args:
            codebook_size (int): The number of codes in the codebook.
            embedding_dim (int): The dimensionality of the embeddings.
            beta (float): The commitment loss weight.
            ema_decay (float): The decay factor for the EMA update of codebook usage.
        """
        super().__init__()
        self.beta = beta
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.ema_decay = ema_decay

        # --- Codebook Embeddings ---
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=embedding_dim**-0.5)
        # The base codebook is not trained directly
        for p in self.embedding.parameters():
            p.requires_grad = False

        # --- Projection Layer ---
        # A linear projection is applied to the codebook before use
        self.embedding_proj = nn.Linear(embedding_dim, embedding_dim)

        # --- EMA Buffer for Codebook Usage ---
        # Initialize the EMA buffer for codebook usage with a uniform distribution.
        # `register_buffer` makes it part of the module's state, but not a trainable parameter.
        initial_usage = torch.ones(codebook_size) / codebook_size
        self.register_buffer("ema_codebook_usage", initial_usage)

    def forward(self, z: torch.Tensor):
        """
        Forward pass of the SimVQ module.

        Args:
            z (torch.Tensor): The input tensor from the encoder, shape `(B, ..., D)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - z_q (torch.Tensor): The quantized output tensor, same shape as z.
            - commit_loss (torch.Tensor): The commitment loss.
            - perplexity (torch.Tensor): The perplexity of the codebook usage for the current batch.
            - ema_entropy (torch.Tensor): The entropy of the EMA of codebook usage.
        """
        assert z.shape[-1] == self.embedding_dim

        # Flatten input tensor to (Batch*H*W, Dim)
        z_flattened = z.view(-1, self.embedding_dim)

        # The actual codebook used for quantization is the projected one
        quant_codebook = self.embedding_proj(self.embedding.weight)

        # Calculate distances: (z - e)^2 = z^2 + e^2 - 2*z*e
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(quant_codebook**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, einops.rearrange(quant_codebook, "n d -> d n"))
        )

        # Find the closest codebook vectors
        min_encoding_indices = torch.argmin(d, dim=1)

        # Quantize the input tensor
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)

        # --- Update EMA and Calculate Metrics ---

        # 1. Calculate the empirical distribution for the current batch
        codebook_counts = torch.bincount(min_encoding_indices, minlength=self.codebook_size)
        e_mean = codebook_counts.float() / min_encoding_indices.numel()

        # 2. Update the EMA of codebook usage (in-place)
        with torch.no_grad():
            self.ema_codebook_usage.mul_(self.ema_decay).add_(e_mean, alpha=1 - self.ema_decay)

        # 3. Calculate the entropy of the EMA distribution: H(p) = -Σ p_i * log(p_i)
        ema_entropy = -torch.sum(self.ema_codebook_usage * torch.log(self.ema_codebook_usage + 1e-10))

        # 4. Calculate perplexity for the current batch
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # --- Loss Calculation ---
        commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)

        # Use straight-through estimator for gradients
        z_q = z + (z_q - z).detach()

        return z_q, commit_loss, perplexity, ema_entropy


class VanillaVQ(nn.Module):
    def __init__(self, codebook_size: int = 8192, embedding_dim: int = 128, beta: float = 0.25):
        super().__init__()
        self.beta = beta
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / embedding_dim, 1.0 / embedding_dim)

    def forward(self, z: torch.Tensor):
        assert z.shape[-1] == self.embedding_dim

        z_flattened = z.view(-1, self.embedding_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight**2, dim=1)
            - 2 * torch.einsum("bd,dn->bn", z_flattened, einops.rearrange(self.embedding.weight, "n d -> d n"))
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        codebook_counts = torch.bincount(min_encoding_indices, minlength=self.codebook_size)
        e_mean = codebook_counts.float() / min_encoding_indices.numel()
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-8)))

        commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()

        return z_q, commit_loss, perplexity


class FSQ(nn.Module):
    """
    Finite Scalar Quantization (FSQ) - Final Corrected Implementation
    
    This implementation strictly follows the user's specified procedure:
    tanh -> scale by L/2 -> floor -> STE on round(floor(z)).
    """
    def __init__(self, levels: list[int]):
        """
        Initializes the FSQ module.
        
        Args:
            levels (list[int]): A list of integers specifying the number of
                                quantization levels for each dimension.
        """
        super().__init__()
        
        self.embedding_dim = len(levels)
        self.codebook_size = math.prod(levels)
        
        # Register levels as a buffer for automatic device placement
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float))
        
        # Pre-calculate the offset for index conversion. For levels L, indices are
        # calculated from quantized values by adding an offset of floor(L/2).
        self.register_buffer("index_offset", torch.floor(self.levels / 2))
        
        # Pre-compute bases for converting multi-dimensional indices to a single flat index.
        # This is used for perplexity calculation.
        _bases = [math.prod(levels[i+1:]) for i in range(len(levels)-1)] + [1]
        self.register_buffer("_bases", torch.tensor(_bases, dtype=torch.long))

    def forward(self, z: torch.Tensor):
        """
        Quantizes the input tensor z following the specified steps.
        
        Args:
            z (torch.Tensor): The input tensor from the encoder, shape (..., embedding_dim).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - z_q (torch.Tensor): The quantized tensor with STE, shape is the same as z.
            - indices (torch.Tensor): The integer indices of the quantized vectors.
            - perplexity (torch.Tensor): A measure of codebook usage.
        """
        # Step 1: Dimension Check
        assert z.shape[-1] == self.embedding_dim, \
            f"Input embedding dim {z.shape[-1]} must match FSQ dim {self.embedding_dim}"
        
        # Step 2: Tanh Normalization
        z_tanh = torch.tanh(z)
        
        # Step 3: Scale by L/2
        x = z_tanh * torch.floor(self.levels / 2)

        # Step 5: Round x for quantization target and apply STE
        x_quantized = torch.round(x) # This is the actual discrete quantized value
        z_q = x + (x_quantized - x).detach() # The STE formula you specified
        
        z_q = z_q / self.levels  # Normalize back to the range [0, 1] for each dimension
        
        # Step 6: Calculate indices from the quantized values
        # Add offset to make indices non-negative
        indices_unclamped = (x_quantized + self.index_offset).long()
        
        # Clamp indices to the valid range [0, L-1] for each dimension
        zeros = torch.zeros_like(self.levels, dtype=torch.long)
        max_indices = (self.levels - 1).long()
        indices_per_dim = torch.max(torch.min(indices_unclamped, max_indices), zeros)
        
        # --- Metrics Calculation ---
        
        # Step 7: Calculate Perplexity
        # Flatten spatial/batch dimensions for index and perplexity calculation
        original_shape = indices_per_dim.shape
        indices_per_dim_flat = indices_per_dim.view(-1, self.embedding_dim)

        # Convert multi-dimensional indices to a single flat index
        flat_indices = torch.sum(indices_per_dim_flat * self._bases, dim=1)
        
        codebook_counts = torch.bincount(flat_indices, minlength=self.codebook_size)
        e_mean = codebook_counts.float() / flat_indices.numel()
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-8)))
        
        # Reshape indices to have the same non-channel dimensions as the input
        indices = flat_indices.view(*original_shape[:-1])

        return z_q, indices, perplexity


class CosVQ(nn.Module):
    def __init__(self, codebook_size: int = 8192, embedding_dim: int = 128, beta: float = 0.25, temperature: float = 0.1):
        super().__init__()
        self.beta = beta
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / embedding_dim, 1.0 / embedding_dim)

    def forward(self, z: torch.Tensor):
        assert z.shape[-1] == self.embedding_dim, "Input last dimension must match embedding_dim"

        original_z_shape = z.shape
        z_flattened = z.view(-1, self.embedding_dim)  # (B*H*W, D)

        z_norm = F.normalize(z_flattened, p=2, dim=1)  # (B*H*W, D)
        embedding_norm = F.normalize(self.embedding.weight, p=2, dim=1)  # (N, D)

        cos_sim_scores = torch.matmul(z_norm, embedding_norm.t())
        min_encoding_indices = torch.argmax(cos_sim_scores, dim=1)  # (B*H*W,)
        z_q = self.embedding(min_encoding_indices).view(original_z_shape)

        codebook_counts = torch.bincount(min_encoding_indices, minlength=self.codebook_size)
        e_mean = codebook_counts.float() / min_encoding_indices.numel()
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-8)))

        commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2) + torch.mean((z_q - z.detach()) ** 2)
        z_q = z + (z_q - z).detach()

        p_logits = cos_sim_scores / self.temperature
        log_probs = F.log_softmax(p_logits, dim=1)
        probs = torch.exp(log_probs)  # (B*H*W, N)
        P_avg = torch.mean(probs, dim=0)  # 形状 (N,)
        P_avg = P_avg + 1e-8
        entropy_loss = -torch.sum(P_avg * torch.log(P_avg))

        return z_q, commit_loss, perplexity, entropy_loss


class CosVQ_EMA(nn.Module):
    def __init__(self, codebook_size: int = 8192, embedding_dim: int = 128, beta: float = 0.25, temperature: float = 0.1, ema_decay: float = 0.8):
        super().__init__()
        self.beta = beta
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.ema_decay = ema_decay

        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / embedding_dim, 1.0 / embedding_dim)

        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_embedding_data", torch.zeros(codebook_size, embedding_dim))

    def forward(self, z: torch.Tensor):
        assert z.shape[-1] == self.embedding_dim, "Input last dimension must match embedding_dim"

        original_z_shape = z.shape
        z_flattened = z.view(-1, self.embedding_dim)  # (B*H*W, D)

        z_norm = F.normalize(z_flattened, p=2, dim=1)  # (B*H*W, D)
        embedding_norm = F.normalize(self.embedding.weight, p=2, dim=1)  # (N, D)

        cos_sim_scores = torch.matmul(z_norm, embedding_norm.t())  # (B*H*W, N)
        min_encoding_indices = torch.argmax(cos_sim_scores, dim=1)  # 形状 (B*H*W,)

        z_q = self.embedding(min_encoding_indices).view(original_z_shape)

        with torch.no_grad():
            min_encoding_onehot = F.one_hot(min_encoding_indices, num_classes=self.codebook_size).float()
            self.cluster_size.data.mul_(self.ema_decay).add_(min_encoding_onehot.sum(0), alpha=1 - self.ema_decay)
            self.ema_embedding_data.data.mul_(self.ema_decay).add_(torch.matmul(min_encoding_onehot.t(), z_flattened), alpha=1 - self.ema_decay)
            normalized_ema_embedding_data = self.ema_embedding_data / self.cluster_size.unsqueeze(1).clamp(min=1e-5)
            self.embedding.weight.data.copy_(normalized_ema_embedding_data)

        codebook_counts = torch.bincount(min_encoding_indices, minlength=self.codebook_size)
        e_mean = codebook_counts.float() / min_encoding_indices.numel()
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-8)))

        commit_loss = F.mse_loss(z_flattened, z_q.detach().view(-1, self.embedding_dim))
        commit_loss = self.beta * commit_loss

        z_q_ste = z + (z_q - z).detach()

        p_logits = cos_sim_scores / self.temperature
        log_probs = F.log_softmax(p_logits, dim=1)
        probs = torch.exp(log_probs)
        P_avg = torch.mean(probs, dim=0)
        P_avg = P_avg + 1e-8
        entropy_loss = -torch.sum(P_avg * torch.log(P_avg))

        return z_q_ste, commit_loss, perplexity, entropy_loss


class CosVQ_Reactivation(nn.Module):
    def __init__(
        self, codebook_size: int = 8192, embedding_dim: int = 128, beta: float = 0.25, temperature: float = 0.1, prob_ema_decay: float = 0.9
    ):
        super().__init__()
        self.beta = beta
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.prob_ema_decay = prob_ema_decay  # EMA decay specifically for the probability table

        # Codebook embeddings: This remains a standard nn.Embedding
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / embedding_dim, 1.0 / embedding_dim)

        self.register_buffer("codebook_probs_ema", torch.full((codebook_size,), 1.0 / codebook_size))

    def forward(self, z: torch.Tensor):
        assert z.shape[-1] == self.embedding_dim, "Input last dimension must match embedding_dim"

        original_z_shape = z.shape
        z_flattened = z.view(-1, self.embedding_dim)  # Reshape z to (B*H*W, D) for processing

        # Normalize z and embeddings for cosine similarity calculation
        z_norm = F.normalize(z_flattened, p=2, dim=1)  # L2 normalize input vectors
        embedding_norm = F.normalize(self.embedding.weight, p=2, dim=1)  # L2 normalize codebook vectors
        cos_sim_scores = torch.matmul(z_norm, embedding_norm.t())
        min_encoding_indices = torch.argmax(cos_sim_scores, dim=1)  # Shape: (B*H*W,)
        z_q = self.embedding(min_encoding_indices).view(original_z_shape)

        codebook_counts = torch.bincount(min_encoding_indices, minlength=self.codebook_size)
        e_mean = codebook_counts.float() / min_encoding_indices.numel()
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-8)))  # Add epsilon for numerical stability

        commit_loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)

        z_q_ste = z + (z_q - z).detach()

        p_logits_for_entropy = cos_sim_scores / self.temperature
        log_probs_for_entropy = F.log_softmax(p_logits_for_entropy, dim=1)
        probs_for_entropy = torch.exp(log_probs_for_entropy)  # (B*H*W, N)
        P_avg_for_entropy = torch.mean(probs_for_entropy, dim=0)  # (N,)

        P_avg_for_entropy = P_avg_for_entropy + 1e-8  # Add epsilon for stability
        entropy_loss = -torch.sum(P_avg_for_entropy * torch.log(P_avg_for_entropy))

        with torch.no_grad():
            P_avg_current_batch = P_avg_for_entropy.data  # Detach for EMA update
            self.codebook_probs_ema.data.mul_(self.prob_ema_decay).add_(e_mean, alpha=1 - self.prob_ema_decay)
            threshold = 0.0125 / self.codebook_size
            dead_codes_mask = self.codebook_probs_ema < threshold  # Boolean mask indicating dead codes
            num_dead_codes = dead_codes_mask.sum().item()
            if num_dead_codes > 0:
                log.info(f"Reactivating {num_dead_codes} dead codes")
                dead_code_indices = torch.where(dead_codes_mask)[0]  # Get the indices of dead codes
                num_samples_from_batch = min(num_dead_codes, z_flattened.shape[0])
                rand_z_indices = torch.randperm(z_flattened.shape[0])[:num_samples_from_batch]
                self.embedding.weight.data[dead_code_indices[:num_samples_from_batch]] = z_flattened[rand_z_indices].to(self.embedding.weight.dtype)
                self.codebook_probs_ema.data[dead_code_indices[:num_samples_from_batch]] = 1.0 / self.codebook_size

        return z_q_ste, commit_loss, perplexity, entropy_loss, self.codebook_probs_ema.min()


if __name__ == "__main__":
    device = "cuda:1"

    m = SimVQ(codebook_size=8192, embedding_dim=128, beta=0.25)
    m = m.to(device)
    x = torch.randn((2, 8, 128), device=device)
    x.requires_grad_(True)

    x_q, commit_loss, min_encoding_indices = m(x)
