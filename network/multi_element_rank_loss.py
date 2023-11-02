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

        print("config_idxs: ", config_idxs)
        print("permuted_idxs: ", permuted_idxs)
        # We mask those cases where we compare the same configuration
        config_mask = torch.where(config_idxs != permuted_idxs, 1, 0)
        
        print("config_mask: ", config_mask)
        print("config_runtime: ", config_runtime)
        permuted_runtime = config_runtime[:, permutation]
        print("permuted_runtime: ", permuted_runtime)
        labels = 2*((config_runtime - permuted_runtime) > 0) -1
        permuted_output = outputs[:, permutation]

        print("outputs: ", outputs.view(-1,1))
        print("permuted_output: ", permuted_output.view(-1,1))
        print("labels: ", labels.view(-1,1))
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
            [   7,   10,    6,   10,    9,    0,    4,    6,   11,    6,    0,    7,
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

    outputs=torch.tensor([[ 3.13,  3.12,  3.14, ],
            ],
        )

    config_runtime=torch.Tensor([[ 0.13,  0.12,  0.14, ],],
        )

    config_idxs=torch.Tensor([[ 128, 110, 50],
            ],
        ).long()

    """
    1) If all the `config_idxs` are the same, then we will get a zero loss
    2) If all the `config_runtime` are the same, then loss will not be zero and depdending on the outputs, it can take any value
    """

    merl = MultiElementRankLoss()
    loss = merl(outputs=outputs, config_runtime=config_runtime, config_idxs=config_idxs)
    
    print(loss)