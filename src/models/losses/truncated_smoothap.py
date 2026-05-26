# Implemented in MinkLoc3Dv2

import numpy as np
import torch

from src.models.losses.loss_utils import sigmoid, compute_aff


class TruncatedSmoothAP:
    """Truncated Smooth-AP loss with top-k positives per query."""
    def __init__(self, tau1: float = 0.01, similarity: str = 'cosine', positives_per_query: int = 4,
                 # Hard negative mining parameters (backward compatible - disabled by default)
                 use_hard_negatives: bool = False, hard_negative_ratio: float = 0.3,
                 tau2: float = 0.02, margin: float = 0.2):
        self.tau1 = tau1
        self.similarity = similarity
        self.positives_per_query = positives_per_query
        # Hard negative mining parameters
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_ratio = hard_negative_ratio
        self.tau2 = tau2  # Temperature for hard negative sigmoid
        self.margin = margin  # Margin for hard negative selection

    def __call__(self, embeddings, positives_mask, negatives_mask):
        device = embeddings.device
        positives_mask = positives_mask.to(device)
        negatives_mask = negatives_mask.to(device)

        # Pairwise similarity: rows = queries, cols = items
        s_qz = compute_aff(embeddings, similarity=self.similarity)

        # Count positives per query
        n_positives = positives_mask.sum(dim=1)  # (batch_size,)
        
        # Check if any query has no positives
        if n_positives.min() == 0:
            # Fallback: return zero loss if any query has no positives
            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
            stats = {
                'positives_per_query': n_positives.float().mean().item(),
                'best_positive_ranking': 0.0,
                'recall': {1: 0.0},
                'loss': 0.0,
                'ap': 0.0,
                'avg_embedding_norm': embeddings.norm(dim=1).mean().item()
            }
            return dummy_loss, stats
        
        # Select top-k positives per query (adapt k to minimum available positives)
        s_positives = s_qz.detach().clone()
        s_positives.masked_fill_(~positives_mask, np.NINF)
        
        # Use the minimum of requested k and minimum positives in batch
        # This ensures all queries have enough positives for topk
        min_positives_in_batch = n_positives.min().item()
        k_to_use = min(self.positives_per_query, min_positives_in_batch)
        k_to_use = max(1, k_to_use)  # At least 1 to avoid errors
        
        closest_positives_ndx = torch.topk(
            s_positives, k=k_to_use, dim=1, largest=True, sorted=True
        )[1]  # (batch_size, k_to_use)

        # Smooth ranks
        s_diff = s_qz.unsqueeze(1) - s_qz.gather(1, closest_positives_ndx).unsqueeze(2)
        s_sigmoid = sigmoid(s_diff, temp=self.tau1)

        # Numerator: rank among positives (exclude the exact positive position)
        pos_mask = positives_mask.unsqueeze(1)
        pos_s_sigmoid = s_sigmoid * pos_mask
        mask = torch.ones_like(pos_s_sigmoid).scatter(2, closest_positives_ndx.unsqueeze(2), 0.)
        pos_s_sigmoid = pos_s_sigmoid * mask
        r_p = torch.sum(pos_s_sigmoid, dim=2) + 1.0

        # Denominator: rank among positives + negatives
        neg_mask = negatives_mask.unsqueeze(1)
        neg_s_sigmoid = s_sigmoid * neg_mask
        r_omega = r_p + torch.sum(neg_s_sigmoid, dim=2)

        r = r_p / r_omega  # (N, k)

        # Metrics
        stats = {}
        stats['positives_per_query'] = n_positives.float().mean().item()
        temp = (s_diff.detach() > 0)
        temp = torch.logical_and(temp[:, 0], negatives_mask)
        hard_ranking = temp.sum(dim=1)
        stats['best_positive_ranking'] = hard_ranking.float().mean().item()
        stats['recall'] = {1: (hard_ranking <= 1).float().mean().item()}

        # Average precision over available positives
        valid_positives_mask = torch.gather(positives_mask, 1, closest_positives_ndx)
        masked_r = r * valid_positives_mask
        n_valid_positives = valid_positives_mask.sum(dim=1)
        valid_q_mask = n_valid_positives > 0
        masked_r = masked_r[valid_q_mask]

        ap = (masked_r.sum(dim=1) / n_valid_positives[valid_q_mask]).mean()

        # Hard negative mining (optional, backward compatible)
        if self.use_hard_negatives and negatives_mask.sum() > 0:
            # Get negative similarities
            neg_similarities = s_qz * negatives_mask.float()

            # For each query, select hard negatives (high similarity negatives)
            num_negatives_per_query = negatives_mask.sum(dim=1)
            num_hard_per_query = (num_negatives_per_query.float() * self.hard_negative_ratio).long()
            max_hard = num_hard_per_query.max().item()

            if max_hard > 0:
                # Get top-k hardest negatives for each query
                hard_neg_values, hard_neg_indices = torch.topk(
                    neg_similarities, k=min(max_hard, neg_similarities.size(1)),
                    dim=1, largest=True, sorted=True
                )

                # Create mask for hard negatives
                hard_neg_mask = torch.zeros_like(negatives_mask).float()
                for i in range(hard_neg_mask.size(0)):
                    if num_hard_per_query[i] > 0:
                        hard_neg_mask[i, hard_neg_indices[i, :num_hard_per_query[i]]] = 1.0

                # Compute hard negative loss with different temperature (tau2)
                s_diff_hard = s_qz.unsqueeze(1) - s_qz.gather(1, closest_positives_ndx).unsqueeze(2)
                s_sigmoid_hard = sigmoid(s_diff_hard, temp=self.tau2)

                # Rank computation for hard negatives
                hard_neg_s_sigmoid = s_sigmoid_hard * hard_neg_mask.unsqueeze(1)
                r_hard = r_p / (r_p + torch.sum(hard_neg_s_sigmoid, dim=2))

                # Combine original AP loss with hard negative loss
                hard_ap = (r_hard * valid_positives_mask).sum(dim=1)
                hard_ap = hard_ap[valid_q_mask] / n_valid_positives[valid_q_mask]
                hard_ap = hard_ap.mean()

                # Weighted combination of losses
                loss = 0.7 * (1.0 - ap) + 0.3 * (1.0 - hard_ap)
                stats['hard_neg_ratio'] = self.hard_negative_ratio
                stats['hard_ap'] = hard_ap.item()
            else:
                loss = 1.0 - ap
        else:
            # Original loss (backward compatible)
            loss = 1.0 - ap

        stats['loss'] = loss.item()
        stats['ap'] = ap.item()
        stats['avg_embedding_norm'] = embeddings.norm(dim=1).mean().item()
        return loss, stats
