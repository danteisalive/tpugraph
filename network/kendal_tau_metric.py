import torch
from scipy.stats import kendalltau
from torch import nn
import torchmetrics as tm
import math
class KendallTau(tm.Metric):

    higher_is_better = True

    def __init__(self,) -> None:
        super().__init__()
        self.add_state("runtimes", default=[], dist_reduce_fx=None)

    def update(self, 
               preds: torch.Tensor, # (bs, num_configs)
               target: torch.Tensor, # (bs, num_configs)
               selected_configs : torch.Tensor = None, # (bs, num_configs)
               ) -> None:
        """
        Update the metric state
        Args:
            preds: Tensor of shape (bs, num_configs) with the predicted runtimes orders
            target: Tensor of shape (bs, num_configs) with the target runtimes
        """
        predicted_rankings = preds.cpu().numpy()
        actual_rankings = target.cpu().numpy()

        # print(predicted_rankings.shape, actual_rankings.shape, selected_configs.shape)
        # print("predicted_rankings: ", predicted_rankings,)
        # print("actual_rankings: ", actual_rankings)
        # print("selected_configs: ", selected_configs)

        kts = []
        for idx in range(len(preds)):
            corr, _ = kendalltau(predicted_rankings[idx], actual_rankings[idx])
            corr = 0 if math.isnan(corr) else corr
            # print(corr)
            kts.append(corr)

        self.runtimes.append(torch.Tensor(kts))


    def compute(self) -> torch.Tensor:
        return torch.cat(self.runtimes).mean()
    
    def dump(self) -> torch.Tensor:
        print(self.runtimes)