"""
Implements the Jensen Shannon divergence for training distribution neural networks.
"""

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from mtm.models.protonet.model import (
    get_prototypes,
    gaussian_prototypical_mahalanobis_loss,
)


def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """
    Estimates covariance matrix like numpy.cov
    https://github.com/pytorch/pytorch/issues/19037
    """
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w / w_sum)[:, None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


def cov_batch(embeddings, n_classes, k_shots_per_class):
    # (batch_size, num_examples, embedding_size)
    batch_size, embedding_size = embeddings.size(0), embeddings.size(-1)
    embeddings_reshaped = embeddings.reshape(
        [batch_size, n_classes, k_shots_per_class, embedding_size]
    )
    covs = []
    for i in range(embeddings.shape[0]):
        covs_n = []
        for j in range(n_classes):
            covs_n.append(cov(embeddings_reshaped[i, j, :, :]))
        covs.append(torch.stack(covs_n))

    return torch.stack(covs)


def jsd_loss(
    support_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    query_targets: torch.Tensor,
    n_classes: int,
    k_shots_per_class: int,
    test_shots_per_class: int,
    generalized_jsd_over_supports: bool = True,
    mahala: bool = True,
):
    """
    Computes the Jensen Shannon divergence (JSD) between the support and query embeddings plus the negative generalized JSD over supports:
        L = JSD(query, support) - 1/n * sum([DKL(P_i || P_bar) for P_i in P])
            where P_i is the distribution of embeddings for class i and P_bar is the average embedding over all classes
    :param support_embeddings:
    :param query_embeddings:
    :param n_classes:
    :param k_shots_per_class:
    :param mahal: if True, use cross entropy of Mahalanobis distance instead of average JSD between query and support
    :param generalized_jsd_over_supports: if True subtract from the loss the generalized JSD. This will force the network to maximize divergence between class distributions in addition to minimizing divergence between supports and queries.
    :return:
    """
    prototypes = get_prototypes(
        support_embeddings, n_classes, k_shots_per_class, return_sd=False
    )

    loss_terms = {}

    if mahala:  # TODO: refactor mahalanobis out of jsd_loss
        loss = gaussian_prototypical_mahalanobis_loss(
            prototypes, embeddings=query_embeddings, targets=query_targets
        )
        loss_terms["Mahalanobis"] = loss.item()
        # print(f"Mahalanobis loss {loss.item()}")
    else:
        embedding_size = prototypes.shape[-1]
        support_means = prototypes[:, :, 0 : embedding_size // 2]
        support_sds = prototypes[:, :, embedding_size // 2 :]
        support_sds = support_sds.mul(0.5).exp_()
        support_distributions = [
            MultivariateNormal(
                support_means[:, i, :], torch.diag_embed(support_sds[:, i])
            )
            for i in range(n_classes)
        ]
        query_prototypes = get_prototypes(
            query_embeddings, n_classes, test_shots_per_class, return_sd=False
        )
        query_means = query_prototypes[:, :, 0 : embedding_size // 2]
        query_sds = query_prototypes[:, :, embedding_size // 2 :]
        query_sds = query_sds.mul(0.5).exp_()
        # Represent distributions as Gaussians
        query_distributions = [
            MultivariateNormal(query_means[:, i, :], torch.diag_embed(query_sds[:, i]))
            for i in range(n_classes)
        ]
        # mixture of support and query:
        ms = [
            MultivariateNormal(
                0.5 * (support_means[:, i, :] + query_means[:, i, :]),
                torch.diag_embed(0.5 * (support_sds[:, i] + query_sds[:, i])),
            )
            for i in range(n_classes)
        ]

        # Compute the average JSD between query and support examples over all classes
        loss = 0
        for p, q, m in zip(support_distributions, query_distributions, ms):
            loss += (1.0 / n_classes) * (
                1
                / 2.0
                * (
                    torch.distributions.kl.kl_divergence(p, m)
                    + torch.distributions.kl.kl_divergence(q, m)
                )
            )
        # print(f"average loss between query and support distributions {loss.item()}")
        loss_terms["JSD_between_query_and_support"] = loss.item()

    if generalized_jsd_over_supports:
        embedding_size = prototypes.shape[-1]
        support_means = prototypes[:, :, 0 : embedding_size // 2]
        support_sds = prototypes[:, :, embedding_size // 2 :]
        support_sds = support_sds.mul(0.5).exp_()
        support_distributions = [
            MultivariateNormal(
                support_means[:, i, :], torch.diag_embed(support_sds[:, i, :])
            )
            for i in range(n_classes)
        ]
        # And generalized JSD over supports
        # TODO: note in paper the computational advantage of generalized JSD over pairwise kl divergences (which grows exponentially in the number of classes)
        p_bar = MultivariateNormal(
            support_means.mean(1), torch.diag_embed(support_sds.mean(1))
        )
        generalized_jsd = (
            1.0
            / torch.tensor(
                [
                    torch.distributions.kl.kl_divergence(p, p_bar)
                    for p in support_distributions
                ]
            ).mean()
        )

        loss_terms["generalized_JSD_over_supports"] = generalized_jsd.item()
        # print(f"Average generalized loss: {generalized_jsd.item()}")
        # Add to the loss
        loss += generalized_jsd

    return prototypes, loss, loss_terms
