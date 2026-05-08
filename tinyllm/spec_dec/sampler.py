import torch


def greedy_accept(
    draft_tokens: torch.Tensor,  # [K]
    logits: torch.Tensor,        # [K+1, vocab_size]
) -> tuple[torch.Tensor, int]:
    K = draft_tokens.shape[0]
    accepted: list[torch.Tensor] = []

    for i in range(K):
        target_tok = int(logits[i].argmax())
        draft_tok = int(draft_tokens[i])
        if target_tok == draft_tok:
            accepted.append(draft_tokens[i])
        else:
            # Target disagrees — use its prediction, discard rest of draft.
            accepted.append(torch.tensor(target_tok, device=draft_tokens.device))
            return torch.stack(accepted), len(accepted) - 1

    # All K accepted — collect the bonus from the last target logit.
    bonus = logits[K].argmax()
    accepted.append(bonus)
    return torch.stack(accepted), K
