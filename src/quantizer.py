import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.codebook_size = config["model"]["codebook_size"]
        self.emb_dim = config["model"]["emb_dim"]
        self.gamma = config["model"]["ema_gamma"]
        self.warmup = config["model"]["dead_code_warmup"]
        self.update_every = config["model"]["dead_code_replace_every"]
        self.dead_code_threshold = config["model"]["dead_code_threshold"]
        self.register_buffer("inited", torch.zeros((), dtype=torch.long))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))
        self.register_buffer("freq", torch.zeros(self.codebook_size))
        self.register_buffer("sums", torch.zeros(self.codebook_size, self.emb_dim))
        self.register_buffer("codebook", torch.zeros(self.codebook_size, self.emb_dim))

    def select_nearest(self, x):
        dists = x.pow(2).sum(dim=1, keepdim=True) - 2 * x @ self.codebook.t() + self.codebook.pow(2).sum(dim=1)
        return dists.argmin(dim=1)

    @torch.no_grad()
    def update_ema(self, x, inds):
        one_hot = F.one_hot(inds, self.codebook_size).float()

        batch_freq = one_hot.sum(dim=0)
        batch_sum = one_hot.T @ x

        self.freq.mul_(self.gamma).add_(batch_freq, alpha=1 - self.gamma)
        self.sums.mul_(self.gamma).add_(batch_sum, alpha=1 - self.gamma)

        d = self.freq.clamp_min(1e-5).unsqueeze(1)
        self.codebook.copy_(self.sums / d)
        self.updates += 1

    @torch.no_grad()
    def replace_dead_codes(self, x):
        dead = self.freq < self.dead_code_threshold
        num_dead = dead.sum().item()
        if num_dead == 0:
            return
        sz = x.shape[0]
        samples = torch.randint(0, sz, [num_dead], device=x.device)

        msk = x[samples]
        self.codebook[dead] = msk
        self.sums[dead] = msk * self.dead_code_threshold
        self.freq[dead] = self.dead_code_threshold

    @torch.no_grad()
    def calc_stat(self, inds):
        avg_probs = F.one_hot(inds.reshape(-1), self.codebook_size).float().mean(dim=0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())
        usage = (avg_probs > 0).float().mean()

        return {
            "perplexity": perplexity,
            "usage": usage
        }

    @torch.no_grad()
    def initialize(self, x):
        s = int(self.inited.item())
        cnt = min(x.shape[0], self.codebook_size - s)
        if cnt <= 0:
            return

        vals = x[:cnt].detach()
        self.codebook[s:s + cnt].copy_(vals)
        self.freq[s:s + cnt].fill_(self.dead_code_threshold)
        self.sums[s:s + cnt].copy_(vals * self.dead_code_threshold)
        self.inited += cnt

    def forward(self, x):
        b, d, s = x.shape  # d = self.emb_dim
        x = x.transpose(1, 2).reshape(-1, d)
        if self.inited < self.codebook_size and self.training and torch.is_grad_enabled():
            self.initialize(x)

        inds = self.select_nearest(x)
        if self.training and torch.is_grad_enabled():
            self.update_ema(x, inds)
            if self.inited == self.codebook_size and self.updates >= self.warmup and self.updates % self.update_every == 0:
                self.replace_dead_codes(x)

        stat = self.calc_stat(inds)
        res = self.codebook[inds].view(b, s, d).transpose(1, 2).contiguous()
        inds = inds.view(b, s)
        return res, inds, stat


class ResidualVectorQuantizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quants_num = config["model"]["quants_num"]

        self.quantizers = nn.ModuleList([
            VectorQuantizer(config)
            for _ in range(self.quants_num)
        ])

    @torch.no_grad()
    def initialize(self, x):
        res = x
        for q in self.quantizers:
            q.initialize(res.transpose(1, 2).reshape(-1, res.shape[1]))
            y, _, _ = q(res)
            res = res - y

    def forward(self, x):
        perplexities = []
        usages = []

        orig = x
        res = x
        q_sum = torch.zeros_like(x)
        for q in self.quantizers:
            y, inds, stat = q(res)
            q_sum = q_sum + y
            res = res - y.detach()

            perplexities.append(stat["perplexity"])
            usages.append(stat["usage"])

        q_skip = orig + (q_sum - orig).detach()

        return {
            "quantized": q_skip,
            "commitment_loss": F.mse_loss(orig, q_sum.detach()),
            "stats": {
                "perplexity": torch.stack(perplexities).mean(),
                "usage": torch.stack(usages).mean()
            }
        }
