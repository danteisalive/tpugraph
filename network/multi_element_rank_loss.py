import torch
from torch import nn

class MultiElementRankLoss(nn.Module):
    """
    Loss function that compares the output of the model with the output of the model with a permutation of the elements
    """
    
    def __init__(self, margin:float=0.0, number_permutations:int = 1) -> None:
        super().__init__()
        self.loss_fn = nn.MarginRankingLoss(margin=margin, reduction = 'none')
        self.number_permutations = number_permutations
    
    def calculate_rank_loss(self,
                            outputs: torch.Tensor,
                            config_runtime: torch.Tensor,
                            config_idxs: torch.Tensor
                            ):
        """
        Generates a permutation of the predictions and targets and calculates the loss MarginRankingLoss against the permutation
        Args:
            outputs: Tensor of shape (bs, seq_len) with the outputs of the model
            config_runtime: Tensor of shape (bs, seq_len) with the runtime of the model
            config_mask: Tensor of shape (bs, seq_len) with 1 in the positions of the elements
            and 0 in the positions of the padding
        Returns:
            loss: Tensor of shape (bs, seq_len) with the loss for each element in the batch
        """
        bs, num_configs = outputs.shape
        permutation = torch.randperm(num_configs) 
        permuted_idxs = config_idxs[:, permutation]
        # We mask those cases where we compare the same configuration
        config_mask = torch.where(config_idxs != permuted_idxs, 1, 0)
        permuted_runtime = config_runtime[:, permutation]
        labels = 2*((config_runtime - permuted_runtime) > 0) -1
        permuted_output = outputs[:, permutation]
        loss = self.loss_fn(outputs.view(-1,1), permuted_output.view(-1,1), labels.view(-1,1))
        loss = loss.view(bs, num_configs) * config_mask
        return loss.mean()
                
    
    def forward(self,
                outputs: torch.Tensor,
                config_runtime: torch.Tensor,
                config_idxs: torch.Tensor
                ):
        loss = 0 
        for _ in range(self.number_permutations):
            loss += self.calculate_rank_loss(outputs, config_runtime, config_idxs)
        return loss/ self.number_permutations