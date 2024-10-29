import torch
import numpy as np

DEFAULT_EPS=1e-7
def sinkhorn_scaling(mat, mask=None, tol=1e-6, max_iter=50):
    """
    Sinkhorn scaling procedure.
    :param mat: a tensor of square matrices of shape N x M x M, where N is batch size
    :param mask: a tensor of masks of shape N x M
    :param tol: Sinkhorn scaling tolerance
    :param max_iter: maximum number of iterations of the Sinkhorn scaling
    :return: a tensor of (approximately) doubly stochastic matrices
    """
    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)
        mat = mat.masked_fill(mask[:, None, :] & mask[:, :, None], 1.0)

    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=DEFAULT_EPS)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=DEFAULT_EPS)

        if torch.max(torch.abs(mat.sum(dim=2) - 1.)) < tol and torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol:
            break

    if mask is not None:
        mat = mat.masked_fill(mask[:, None, :] | mask[:, :, None], 0.0)

    return mat


def deterministic_neural_sort(s, tau, mask, dev):
    """
    Deterministic neural sort.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :return: approximate permutation matrices of shape [batch_size, slate_length, slate_length]
    """

    n = s.size()[1]
    one = torch.ones((n, 1), dtype=torch.float32, device=dev)
    s = s.masked_fill(mask[:, :, None], -1e8)
    A_s = torch.abs(s - s.permute(0, 2, 1))
    A_s = A_s.masked_fill(mask[:, :, None] | mask[:, None, :], 0.0)

    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))

    temp = [n - m + 1 - 2 * (torch.arange(n - m, device=dev) + 1) for m in mask.squeeze(-1).sum(dim=1)]
    temp = [t.type(torch.float32) for t in temp]
    temp = [torch.cat((t, torch.zeros(n - len(t), device=dev))) for t in temp]
    scaling = torch.stack(temp).type(torch.float32).to(dev)  # type: ignore

    s = s.masked_fill(mask[:, :, None], 0.0)
    C = torch.matmul(s, scaling.unsqueeze(-2))

    P_max = (C - B).permute(0, 2, 1)
    P_max = P_max.masked_fill(mask[:, :, None] | mask[:, None, :], -np.inf)
    P_max = P_max.masked_fill(mask[:, :, None] & mask[:, None, :], 1.0)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat


def sample_gumbel(samples_shape, device, eps=1e-10) -> torch.Tensor:
    """
    Sampling from Gumbel distribution.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param samples_shape: shape of the output samples tensor
    :param device: device of the output samples tensor
    :param eps: epsilon for the logarithm function
    :return: Gumbel samples tensor of shape samples_shape
    """
    U = torch.rand(samples_shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def stochastic_neural_sort(s, n_samples, tau, mask, beta=1.0, log_scores=True, eps=1e-10, dev='cuda'):
    """
    Stochastic neural sort. Please note that memory complexity grows by factor n_samples.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    Minor modifications applied to the original code (masking).
    :param s: values to sort, shape [batch_size, slate_length]
    :param n_samples: number of samples (approximations) for each permutation matrix
    :param tau: temperature for the final softmax function
    :param mask: mask indicating padded elements
    :param beta: scale parameter for the Gumbel distribution
    :param log_scores: whether to apply the logarithm function to scores prior to Gumbel perturbation
    :param eps: epsilon for the logarithm function
    :return: approximate permutation matrices of shape [n_samples, batch_size, slate_length, slate_length]
    """

    batch_size = s.size()[0]
    n = s.size()[1]
    s_positive = s + torch.abs(s.min())
    samples = beta * sample_gumbel([n_samples, batch_size, n, 1], device=dev)
    if log_scores:
        s_positive = torch.log(s_positive + eps)

    s_perturb = (s_positive + samples).view(n_samples * batch_size, n, 1)
    mask_repeated = mask.repeat_interleave(n_samples, dim=0)

    P_hat = deterministic_neural_sort(s_perturb, tau, mask_repeated)
    P_hat = P_hat.view(n_samples, batch_size, n, n)
    return P_hat

def __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator=-1):
    mask = y_true == padding_indicator

    y_pred[mask] = float('-inf')
    y_true[mask] = 0.0

    _, indices = y_pred.sort(descending=True, dim=-1)
    return torch.gather(y_true, dim=1, index=indices)

def dcg(y_pred, y_true, ats=None, gain_function=lambda x: torch.pow(2, x) - 1, padding_indicator=-1):
    """
    Discounted Cumulative Gain at k.

    Compute DCG at ranks given by ats or at the maximum rank if ats is None.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param ats: optional list of ranks for DCG evaluation, if None, maximum rank is used
    :param gain_function: callable, gain function for the ground truth labels, e.g. torch.pow(2, x) - 1
    :param padding_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :return: DCG values for each slate and evaluation position, shape [batch_size, len(ats)]
    """
    y_true = y_true.clone()
    y_pred = y_pred.clone()

    actual_length = y_true.shape[1]

    if ats is None:
        ats = [actual_length]
    ats = [min(at, actual_length) for at in ats]

    true_sorted_by_preds = __apply_mask_and_get_true_sorted_by_preds(y_pred, y_true, padding_indicator)

    discounts = (torch.tensor(1) / torch.log2(torch.arange(true_sorted_by_preds.shape[1], dtype=torch.float) + 2.0)).to(
        device=true_sorted_by_preds.device)

    gains = gain_function(true_sorted_by_preds)

    discounted_gains = (gains * discounts)[:, :np.max(ats)]

    cum_dcg = torch.cumsum(discounted_gains, dim=1)

    ats_tensor = torch.tensor(ats, dtype=torch.long) - torch.tensor(1)

    dcg = cum_dcg[:, ats_tensor]

    return dcg