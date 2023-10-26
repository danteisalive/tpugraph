import torch
from torch import nn

class ListMLELoss(nn.Module):
    """
    Loss function that compares the output of the model with the output of the model with a permutation of the elements
    """
    
    PADDED_VALUE_INDICATOR = -1

    def __init__(self, eps:float=1e-10, ) -> None:
        super().__init__()
        self.eps = eps
        
    def _find_same_config_idx(self, 
                             y_true: torch.Tensor,
                             config_idx: torch.Tensor,
                             ) -> torch.Tensor:

        M, N = y_true.shape
        # Iterate through each batch
        for m in range(M):

            # Find unique values in config_idx for the current batch
            unique_values = torch.unique(config_idx[m])
            for value in unique_values:
                # Find indices where current unique value occurs
                indices = (config_idx[m] == value).nonzero(as_tuple=True)[0]

                # If value occurs more than once, set y_true values to -1 at all indices except the first
                if indices.numel() > 1:
                    for idx in indices[1:]:
                        y_true[m, idx] = -1
            
        return y_true


    def calculate_loss(self,
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

        y_true = _y_true
        y_pred = _y_pred
        config_idxs = _config_idxs

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

        return self.calculate_loss(_y_pred=outputs, 
                                   _y_true=config_runtime, 
                                   _config_idxs=config_idxs)
    


if __name__ == '__main__':

    outputs=torch.tensor([[ 0.1220,  0.1356,  0.1899,  0.1888,  0.1837,  0.1139,  0.1219,  0.2582,
            0.2213,  0.1315,  0.1642,  0.2449,  0.1746,  0.2054,  0.1189,  0.1710,
            0.1533,  0.1535,  0.1832,  0.1471,  0.1519,  0.1828,  0.1395,  0.1240],
            [-0.0782,  0.1017, -0.0578, -0.0018,  0.0054, -0.1335,  0.2586, -0.1748,
            0.1465,  0.1594,  0.2188, -0.1041, -0.2127,  0.1670, -0.0757,  0.2310,
            -0.1717, -0.1883, -0.2119,  0.1588, -0.1474, -0.1077,  0.2138, -0.1344],
            [-0.0924,  0.0311, -0.1005, -0.0924, -0.0763, -0.0691,  0.0311,  0.0320,
            -0.0691,  0.0384,  0.0384,  0.0176,  0.0176, -0.0957, -0.0957, -0.0763,
            0.0320,  0.0320, -0.0957, -0.0957,  0.0132,  0.0132, -0.0691,  0.0384],
            [ 0.0317,  0.1347,  0.0487,  0.1347,  0.0430,  0.0905, -0.0045,  0.0487,
            0.1241,  0.0487,  0.0905,  0.0317, -0.1187,  0.0317, -0.0937,  0.0487,
            -0.0045,  0.1347,  0.1248,  0.1347,  0.0317,  0.0819, -0.1296, -0.0045],
            [ 0.0740,  0.0418, -0.0090,  0.0835,  0.0139,  0.1277,  0.0785, -0.0975,
            0.0322, -0.0054, -0.0773,  0.1277,  0.1312,  0.1058,  0.0673, -0.0927,
            0.0300,  0.0264,  0.1277,  0.0893,  0.1320,  0.1044,  0.1267,  0.0169],
            [ 0.0542,  0.1615,  0.1029,  0.1673,  0.0108,  0.1299,  0.1010,  0.1921,
            0.0980,  0.1004,  0.1151,  0.2059,  0.0178,  0.1174,  0.1643,  0.1894,
            0.1397,  0.0627,  0.1672,  0.1640,  0.1606,  0.1910, -0.0128,  0.1961],
            [-0.0543, -0.0267, -0.0861,  0.0524,  0.1475,  0.0436, -0.1432, -0.1203,
            0.0761,  0.1112,  0.0534,  0.2108, -0.1675,  0.0188,  0.0208,  0.0749,
            -0.0218,  0.0147, -0.0963, -0.0674, -0.1400, -0.1096, -0.0508, -0.1523],
            [ 0.1845,  0.0857,  0.0562,  0.0659,  0.0578,  0.0591,  0.1511,  0.1230,
            0.0566,  0.0548,  0.0599,  0.0669,  0.0599,  0.0681,  0.0567,  0.0711,
            0.0593,  0.0604,  0.0624,  0.0549,  0.0689,  0.0506,  0.0645,  0.0774]],
        )

    config_runtime=torch.Tensor([[ 1.3588,  3.6458, 13.6581,  2.4953,  3.8045,  1.5076,  1.6132,  6.4341,
            6.7345,  4.7562,  4.9863,  3.0053,  2.0716,  6.8894,  2.9343, 14.0992,
            3.3607,  2.1122,  3.1004, 16.1088, 10.2129,  1.6278,  1.5560,  2.1915],
            [ 2.6015, 11.6338,  1.7098,  2.0006,  2.1041,  8.9756, 17.6682,  7.3404,
            3.3523, 15.1408, 23.2232,  4.6192, 11.7856,  3.5454,  2.2450, 14.5028,
            10.2947,  2.0952,  7.3344,  3.8629,  8.0616,  5.3711, 29.9888,  5.5531],
            [ 1.3899,  0.9730,  1.4525,  1.3899,  1.1641,  1.1325,  0.9730,  0.9862,
            1.1325,  1.0000,  1.0000,  0.9883,  0.9883,  1.4104,  1.4104,  1.1641,
            0.9862,  0.9862,  1.4104,  1.4104,  1.0005,  1.0005,  1.1325,  1.0000],
            [ 1.3488,  1.0066,  1.3591,  1.0066,  1.3865,  1.0000,  1.5319,  1.3591,
            1.1247,  1.3591,  1.0000,  1.3488,  1.4810,  1.3488,  2.6907,  1.3591,
            1.5319,  1.0066,  0.9820,  1.0066,  1.3488,  0.9918,  1.6720,  1.5319],
            [ 1.5840,  1.5983,  1.8573,  1.2316,  1.5414,  1.0000,  1.2796,  2.0015,
            1.6558,  1.7615,  2.1801,  1.0000,  0.9949,  1.2324,  1.4188,  2.1726,
            1.2467,  1.2616,  1.0000,  1.4087,  1.0127,  1.5400,  1.0039,  1.8085],
            [ 7.5591,  1.7123,  1.1915,  1.5849,  4.5340,  1.2448,  5.0064,  3.7584,
            1.8241,  1.3041,  2.3765,  9.8502,  2.6001,  2.3486,  2.5891,  2.1449,
            1.4762, 12.4731,  1.8941,  1.7754,  5.0195,  1.9078,  3.8811,  4.2792],
            [ 2.2364,  3.8292,  7.0460,  2.1175,  2.2781,  1.9522, 14.8199,  2.4344,
            5.1680,  8.4850,  2.8979,  9.3638,  3.7385,  4.2919,  4.4677,  3.6032,
            2.8891,  5.4298,  1.3866,  1.0000,  9.8807,  2.4034,  1.8697,  9.3447],
            [ 2.2158,  1.7310,  3.0919,  4.8690,  2.0040,  5.4094,  1.8375,  2.6378,
            3.1345,  4.8752,  5.0341,  1.7197,  1.7367,  1.0000,  2.2327,  2.5441,
            2.5091,  1.5146,  1.1970,  5.4179,  2.0712,  4.5842,  1.5636,  1.3276]],
        )

    config_idxs=torch.Tensor([[ 141,  112,   41,  168,   69,  166,  253,   28,  190,  211,   22,   71,
            135,   46,  125,    5,   64,  257,  224,  132,   77,  250,  124,  128],
            [1007,  244, 1146, 1269,  383, 1174, 1210, 2987, 1690, 2603,  879, 3253,
            1873,  718,  370, 2119, 3129,  508,  919,  436, 3131, 2620, 2179, 2740],
            [   8,   11,    4,    8,    5,    7,   11,   10,    7,    0,    6,    2,
                2,    3,    3,    5,   10,   10,    3,    3,    1,    1,    7,    6],
            [   10,   10,    6,   10,    9,    0,    4,    6,   11,    6,    0,    7,
                2,    7,    3,    6,    4,   10,    8,   10,    7,    5,    1,    4],
            [ 126,  155,  173,    6,  119,  170,    9,   96,  172,   21,  120,   75,
            53,  178,   45,  135,    3,   99,  113,  109,   55,   12,  168,  158],
            [1421, 1253, 1477, 1268,  160, 1188,  971,  773,  425, 1176,  332,  600,
            721,   21,  543, 1070,  713, 1344,   40,   25,  453,  416,  934,  542],
            [ 482,  963,  717,  464, 1357, 1428, 1926, 2082,  330, 1946, 1672,  538,
            2121, 1609,  256,  397,  249, 2356, 2293, 1554,  741,  758,   53, 2051],
            [ 271,  269,  860,  941,  416,  948,  696, 1027,  234,  215,  301,  780,
            189,    0, 1129,  921,  153,  868,  785, 1049,  284,  101,  843,  985]],
        ).long()


    listmle = ListMLELoss()
    loss = listmle(config_runtime, outputs, config_idxs)
    print(loss)
