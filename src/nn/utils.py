from torch import Tensor
import torch
import yaml
import re


def load_config(path: str) -> dict:
    """Load YAML config and convert scientific notation strings to floats."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    
    def convert_values(obj):
        if isinstance(obj, dict):
            return {k: convert_values(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_values(item) for item in obj]
        elif isinstance(obj, str):
            # Match scientific notation like 1e-3, 2.5e-4, etc.
            if re.match(r'^-?\d+\.?\d*[eE][+-]?\d+$', obj):
                return float(obj)
        return obj
    
    return convert_values(cfg)


def logits_to_pred(logprobs_N_K_C: Tensor, return_binary: bool = False, return_uncertainty: bool = True) -> (Tensor, Tensor):
    """ Get the probabilities/class vector and sample uncertainty from the logits """

    mean_probs_N_C = torch.mean(torch.exp(logprobs_N_K_C), dim=1)
    uncertainty = predictive_entropy(logprobs_N_K_C)

    if not return_binary:
        y_hat = mean_probs_N_C
    else:
        y_hat = torch.argmax(mean_probs_N_C, dim=1)

    if return_uncertainty:
        return y_hat, uncertainty
    else:
        return y_hat


def logit_mean(logprobs_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """ Logit mean with the logsumexp trick - Kirch et al., 2019, NeurIPS """

    return torch.logsumexp(logprobs_N_K_C, dim=dim, keepdim=keepdim) - math.log(logprobs_N_K_C.shape[dim])


def entropy(logprobs_N_K_C: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    """Calculates the Shannon Entropy """

    return -torch.sum((torch.exp(logprobs_N_K_C) * logprobs_N_K_C).double(), dim=dim, keepdim=keepdim)


def mean_sample_entropy(logprobs_N_K_C: Tensor, dim: int = -1, keepdim: bool = False) -> Tensor:
    """Calculates the mean entropy for each sample given multiple ensemble predictions - Kirch et al., 2019, NeurIPS"""

    sample_entropies_N_K = entropy(logprobs_N_K_C, dim=dim, keepdim=keepdim)
    entropy_mean_N = torch.mean(sample_entropies_N_K, dim=1)

    return entropy_mean_N


def predictive_entropy(logprobs_N_K_C: Tensor) -> Tensor:
    """ Computes predictive entropy using ensemble-averaged probabilities """

    return entropy(logit_mean(logprobs_N_K_C, dim=1), dim=-1)


def mutual_information(logprobs_N_K_C: Tensor) -> Tensor:
    """ Calculates the Mutual Information - Kirch et al., 2019, NeurIPS """

    # this term represents the entropy of the model prediction (high when uncertain)
    entropy_mean_N = mean_sample_entropy(logprobs_N_K_C)

    # This term is the expectation of the entropy of the model prediction for each draw of model parameters
    mean_entropy_N = entropy(logit_mean(logprobs_N_K_C, dim=1), dim=-1)

    I = mean_entropy_N - entropy_mean_N

    return I
