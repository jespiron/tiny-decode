import torch
import torch.nn.functional as F


def greedy_accept(
    draft_tokens: torch.Tensor,  # [K]
    logits: torch.Tensor,        # [K+1, vocab_size]
) -> tuple[torch.Tensor, int]:
    K = draft_tokens.shape[0]
    accepted: list[torch.Tensor] = []
    for i in range(K):
        target_tok = int(logits[i].argmax())
        if target_tok == int(draft_tokens[i]):
            accepted.append(draft_tokens[i])
        else:
            accepted.append(torch.tensor(target_tok, device=draft_tokens.device))
            return torch.stack(accepted), len(accepted) - 1
    bonus = logits[K].argmax()
    accepted.append(bonus)
    return torch.stack(accepted), K


def rejection_sample(
    draft_tokens: torch.Tensor,   # [K]
    draft_logprobs: torch.Tensor, # [K, vocab_size]
    logits: torch.Tensor,         # [K+1, vocab_size]
    temperature: float = 1.0,
) -> tuple[torch.Tensor, int]:
    K = draft_tokens.shape[0]

    target_probs = F.softmax(logits / temperature, dim=-1)   # [K+1, vocab]
    draft_probs = draft_logprobs.exp()                        # [K, vocab]

    accepted: list[torch.Tensor] = []

    for i in range(K):
        d = int(draft_tokens[i])
        p_t = target_probs[i, d].item()
        p_d = draft_probs[i, d].item()
        accept_prob = min(1.0, p_t / (p_d + 1e-9))

        if torch.rand(1).item() <= accept_prob:
            accepted.append(draft_tokens[i])
        else:
            # Sample correction from the residual: max(0, p_target - p_draft)
            residual = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
            total = residual.sum()
            if total > 0:
                correction = torch.multinomial(residual / total, num_samples=1).squeeze(0)
            else:
                # Edge case: residual is all-zero — fall back to target argmax.
                correction = target_probs[i].argmax()
            accepted.append(correction)
            return torch.stack(accepted), len(accepted) - 1

    # All K accepted — sample bonus from the last target logit.
    bonus = torch.multinomial(target_probs[K], num_samples=1).squeeze(0)
    accepted.append(bonus)
    return torch.stack(accepted), K
