import torch
from torch import nn

class ListMLELoss(nn.Module):
    """
    Loss function that compares the output of the model with the output of the model with a permutation of the elements
    """
    
    PADDED_VALUE_INDICATOR = -1

    def __init__(self, eps:float=1e-10, 
                    number_permutations:int = 1
                    ) -> None:
        super().__init__()
        self.number_permutations = number_permutations
        self.eps = eps
        
    def _find_same_config_idx(self, 
                             y_true: torch.Tensor,
                             config_idx: torch.Tensor,
                             ):

        # Find unique values in config_idx
        unique_values = torch.unique(config_idx)

        for value in unique_values:
            # Find indices where current unique value occurs
            indices = (config_idx == value).nonzero(as_tuple=True)[1]
            
            # If value occurs more than once, set y_true values to -1 at all indices except the first
            if indices.numel() > 1:
                for idx in indices[1:]:
                    y_true[0, idx] = self.PADDED_VALUE_INDICATOR
        
        return y_true


    def calculate_rank_loss(self,
                            _y_pred: torch.Tensor,
                            _y_true: torch.Tensor,
                            _config_idxs: torch.Tensor
                            ):
        """
            ListMLE loss introduced in "Listwise Approach to Learning to Rank - Theory and Algorithm".
            :param y_pred: predictions from the model, shape [batch_size, slate_length]
            :param y_true: ground truth labels, shape [batch_size, slate_length]
            :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
            :return: loss value, a torch.Tensor
        """
        # y_pred=torch.tensor([[ 0.1220,  0.1356,  0.1899,  0.1888,  0.1837,  0.1139,  0.1219,  0.2582,
        #         0.2213,  0.1315,  0.1642,  0.2449,  0.1746,  0.2054,  0.1189,  0.1710,
        #         0.1533,  0.1535,  0.1832,  0.1471,  0.1519,  0.1828,  0.1395,  0.1240],
        #     ],
        #     )

        # y_true=torch.Tensor([[ 1.3588,  3.6458, 13.6581,  2.4953,  3.8045,  1.5076,  1.6132,  6.4341,
        #         6.7345,  4.7562,  4.9863,  3.0053,  2.0716,  6.8894,  2.9343, 14.0992,
        #         3.3607,  2.1122,  3.1004, 16.1088, 10.2129,  1.6278,  1.5560,  2.1915],
        #         ],
        #     )

        y_true = _y_true
        y_pred = _y_pred
        config_idxs = _config_idxs

        assert y_true.shape[0] == 1, ""
        assert y_pred.shape[0] == 1, ""
        assert config_idxs.shape[0] == 1, ""

        y_true = self._find_same_config_idx(y_true, config_idxs)

        # shuffle for randomised tie resolution
        random_indices = torch.randperm(y_pred.shape[-1])
        y_pred_shuffled = y_pred[:, random_indices]
        y_true_shuffled = y_true[:, random_indices]

        y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

        mask = y_true_sorted == self.PADDED_VALUE_INDICATOR

        preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
        preds_sorted_by_true[mask] = float("-inf")

        max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

        preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

        cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

        observation_loss = torch.log(cumsums + self.eps) - preds_sorted_by_true_minus_max

        observation_loss[mask] = 0.0

        return torch.mean(torch.sum(observation_loss, dim=1))

    def forward(self,
                outputs: torch.Tensor,
                config_runtime: torch.Tensor,
                config_idxs: torch.Tensor
                ):

        loss = 0 
        for _ in range(self.number_permutations):
            loss += self.calculate_rank_loss(y_pred=outputs, y_true=config_runtime, config_idxs=config_idxs)
        return loss/ self.number_permutations