import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from imaginaire.utils import log


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

        commit_loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * torch.mean((z_q - z.detach()) ** 2)
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
