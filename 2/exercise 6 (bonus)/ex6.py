"""
Author: Pascal Pilz
Matr.Nr.: K12111234
Exercise 6
"""

import torch


def ex6(logits: torch.Tensor, activation_function: callable, threshold: torch.Tensor, targets: torch.Tensor):
    if not torch.is_floating_point(logits):
        raise TypeError(f"logits should be a torch tensor of floating points, but is {type(logits)}")
    if not torch.is_tensor(threshold):
        raise TypeError(f"threshold should be a torch tensor, but is {type(threshold)}")
    if not (torch.is_tensor(targets) and targets.dtype == torch.bool):
        raise TypeError(f"targets should be a tensor of datatype torch.bool, "
                        f"but is {type(targets)} with datatype {targets.dtype}")

    if not logits.dim() == targets.dim() == 1:
        raise ValueError(f"logits and targets are multidimensional but should be of shape (n_samples,).\n"
                         f"Got logits: {logits.shape} and targets: {targets.shape} instead.")
    if not logits.shape == targets.shape:
        raise ValueError(f"logits and targets are not of the same shape.\n"
                         f"Got logits: {logits.shape} and targets: {targets.shape} instead.")
    if not (False in targets):
        raise ValueError(f"targets should contain at least once both the values False and True "
                         f"but only contains True.")
    if not (True in targets):
        raise ValueError(f"targets should contain at least once both the values False and True "
                         f"but only contains False.")

    logits = activation_function(logits) >= threshold

    TP = torch.sum(torch.masked_select(logits, targets), dtype=torch.float64)
    TN = torch.sum(torch.masked_select(~logits, ~targets), dtype=torch.float64)
    FP = torch.sum(torch.masked_select(logits, ~targets), dtype=torch.float64)
    FN = torch.sum(torch.masked_select(~logits, targets), dtype=torch.float64)

    ACC = (TP + TN) / (TP + TN + FP + FN)

    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)

    BA = (TPR + TNR) / 2

    try:
        F1 = (2 * TP) / (2 * TP + FP + FN)
    except ZeroDivisionError:
        F1 = 0.0

    return [[int(TP), int(FN)], [int(FP), int(TN)]], float(F1), float(ACC), float(BA)


if __name__ == "__main__":
    input_logits = torch.tensor([-3.8026, -3.1368, -3.3636, -3.8113, -3.6550, -4.0879, -3.7155, -3.4077,
                                 -4.4765, -4.6820, -3.6908, -4.1171, -4.1411, -3.4455, -4.3130, -3.2506,
                                 -3.6323, -3.5907, -3.8836, -3.9973, -3.2776, -4.1793, -3.4327, -3.8649,
                                 -4.9511, -3.8640, -4.0141, -4.6920, -4.4074, -3.1897, -4.4546, -4.3629,
                                 -4.4916, -4.9288, -4.0920, -4.7501, -4.2858, -4.1868, -4.3974, -3.1163,
                                 -3.7967, -3.2943, -3.8693, -3.6450, -3.2359, -3.3349, -3.7322, -3.9573,
                                 -4.6485, -4.9496, -4.9600, -4.1983, -3.5073, -3.8366, -3.1166, -4.7825,
                                 -4.0099, -3.1022, -4.5253, -3.9891, -3.3516, -4.9850, -4.9380, -4.8447,
                                 -3.8028, -3.5858, -3.2094, -3.0995, -3.6200, -3.9332, -3.0684, -4.2924,
                                 -3.2161, -4.1295, -3.3239, -4.0809, -3.5168, -3.8081, -3.7999, -4.2530,
                                 -4.2264, -3.8739, -3.5566, -4.2007, -4.9371, -4.7734, -4.7235, -3.0413,
                                 -4.9445, -3.1757, -4.3991, -4.1333, -3.4071, -3.8965, -4.1287, -4.4207,
                                 -4.0616, -4.8540, -4.2265, -3.7544, -4.5247, -3.6978, -4.8824, -3.4102,
                                 -4.9463, -4.5832, -3.1764, -3.9879, -3.4184, -4.8527, -3.2398, -3.4181,
                                 -4.5545, -4.0713, -4.1280, -3.6379, -4.5262, -3.6983, -3.2151, -3.7145,
                                 -4.7982, -3.0641, -4.6764, -4.7615, -3.0039, -3.4630, -4.0224, -3.1046,
                                 -3.8512, -3.5176, -3.3376, -4.6693, -4.3764, -3.2768, -4.8574, -4.3437,
                                 -3.5979, -4.0800, -4.6339, -3.6869, -3.4381, -3.2801, -3.2854, -4.9481,
                                 -3.4456, -3.4213, -4.2329, -3.8984, -4.3615, -4.0362, -4.1773, -3.7157,
                                 -4.3095, -4.4065, -3.9073, -3.8227, -4.9605, -4.1795, -4.0092, -4.5834,
                                 -4.9478, -3.3974, -4.8810, -4.9653, -4.9451, -4.4298, -3.9705, -4.2599,
                                 -3.2481, -3.0564, -4.6746, -3.1066, -3.8691, -4.5844, -4.7741, -4.4936,
                                 -4.4035, -3.1594, -4.3437, -3.1015, -3.3128, -3.1493, -3.2882, -4.7384,
                                 -3.2022, -4.3237, -4.2057, -3.0608, -4.5121, -3.3965, -4.5053, -3.4998,
                                 -3.0981, -4.8680, -4.9516, -4.9848, -4.9113, -3.3672, -4.3876, -3.7944,
                                 -4.2196, -4.3445, -4.7121, -4.6426, -3.1792, -4.5052, -4.8539, -4.7723,
                                 -4.7028, -3.1249, -3.7925, -4.0322, -4.6735, -3.7044, -3.9191, -3.9608,
                                 -3.0072, -4.7891, -4.1904, -4.3980, -3.3609, -3.1848, -3.4957, -4.7455,
                                 -4.5407, -3.0664, -4.3550, -3.7306, -3.5381, -4.4410, -4.8918, -3.8074,
                                 -3.2628, -4.3727, -3.4900, -3.4065, -4.5866, -4.7513, -3.7533, -3.5859,
                                 -4.7964, -4.8901, -4.4568, -3.7474, -3.5168, -4.7259, -3.1295, -3.4088,
                                 -3.8622, -3.3291, -4.6525, -3.2993, -3.1653, -3.7511, -4.5928, -4.1264,
                                 -3.6967, -4.6637, -4.9207, -4.6577, -4.7813, -3.0497, -4.4575, -3.8342,
                                 -4.8406, -4.9491, -3.5667, -4.3458, -4.5195, -4.6999, -4.6631, -3.8052,
                                 -4.1279, -3.5842, -3.2934, -4.3834, -4.9660, -4.6179, -3.9364, -3.5272,
                                 -3.3781, -3.1362, -4.4831, -4.1107, -3.6959, -3.3267, -3.6058, -3.8142,
                                 -4.9316, -4.1338, -4.1295, -3.5562, -4.9759, -4.3738, -4.1946, -3.4008,
                                 -4.2079, -3.8993, -4.7924, -4.8817, -4.3160, -4.8758, -4.3358, -3.0311,
                                 -3.6767, -4.7923, -4.6220, -3.1809, -4.4228, -3.0198, -3.5842, -4.8856,
                                 -3.5456, -4.0550, -4.4942, -4.7225, -3.9359, -4.2337, -4.0613, -4.2157,
                                 -4.2757, -4.7699, -4.9287, -4.4363, -4.2363, -4.7712, -4.5562, -3.7532,
                                 -3.1025, -3.7025, -3.3752, -3.9203, -4.1036, -3.8019, -4.3488, -3.6894,
                                 -3.7493, -4.1854, -3.9789, -4.6158, -4.1016, -3.9272, -4.6782, -4.8893,
                                 -4.4336, -4.6624, -4.9823, -3.1707, -3.8054, -3.0545, -4.4556, -3.8331,
                                 -4.8001, -3.6642, -4.0325, -4.8445, -4.2181, -3.4943, -4.3473, -4.1857,
                                 -3.2929, -4.7479, -3.8580, -3.5463, -4.2914, -4.5276, -4.1625, -4.2562,
                                 -4.0353, -4.1625, -4.1945, -3.8682, -4.6482, -4.3352, -4.9537, -4.6721,
                                 -3.4643, -4.4867, -4.7356, -3.6027, -4.1991, -3.6227, -4.8120, -3.0157,
                                 -4.0394, -3.1273, -3.4034, -3.9185, -4.4727, -4.3306, -3.6185, -3.3836,
                                 -3.4191, -3.4331, -4.8780, -4.0488, -4.3339, -4.5291, -4.5735, -3.2624,
                                 -3.9148, -3.0020, -3.1801, -3.9744, -3.6090, -4.7608, -4.1373, -3.3027,
                                 -3.4741, -4.5438, -3.7668, -3.0710, -3.8673, -4.3099, -3.2369, -4.3014,
                                 -3.5865, -4.1558, -4.4787, -4.4060, -3.9069, -3.3829, -4.1561, -3.4840,
                                 -3.3404, -3.1971, -3.2292, -3.0757, -3.3736, -3.2531, -3.5252, -3.1763,
                                 -3.2158, -4.1104, -4.4454, -3.9949, -3.6031, -3.5859, -3.2979, -3.0278,
                                 -4.6706, -4.8505, -3.1872, -4.6118, -3.7055, -3.5208, -4.1479, -3.2800,
                                 -3.6935, -3.1027, -3.2907, -4.9370, -3.6810, -4.2242, -4.9556, -4.8462,
                                 -3.4295, -3.8648, -4.2848, -4.5523, -4.2020, -3.2753, -3.5535, -3.9271,
                                 -3.0839, -4.1527, -3.1962, -4.1769, -3.7517, -3.1094, -3.9523, -3.0034,
                                 -3.6704, -3.4306, -3.0359, -4.0771, -4.9679, -3.8233, -3.8627, -3.6736,
                                 -3.5387, -3.2063, -4.9724, -4.8457, -4.2434, -3.9956, -3.2443, -4.0696,
                                 -4.1817, -3.4892, -4.7343, -3.5939, -4.6424, -3.6447, -4.2330, -4.4761,
                                 -4.1697, -3.0294, -3.1995, -4.2327, -4.1099, -3.1144, -3.2458, -4.5531,
                                 -3.2495, -3.7114, -4.6164, -4.4050, -4.1978, -4.8748, -4.6822, -4.2173,
                                 -4.9018, -3.0015, -3.2855, -3.0568, -4.4969, -4.8223, -3.0411, -3.7528,
                                 -3.8306, -4.1634, -4.1700, -4.9672, -4.1943, -3.9798, -3.4361, -3.5600,
                                 -4.1262, -3.5495, -4.2292, -3.7246, -4.0331, -3.6859, -3.7919, -4.6887,
                                 -4.1390, -3.8831, -3.2020, -4.7465, -4.8378, -3.0731, -4.9944, -4.7618,
                                 -4.3488, -4.9944, -4.4625, -3.6074, -3.7536, -4.4533, -3.3804, -3.1207,
                                 -4.4862, -3.7275, -3.9596, -4.1449, -4.1809, -4.2366, -3.8915, -4.7783,
                                 -4.4844, -4.8775, -3.8259, -4.3757, -3.3242, -3.3154, -3.3609, -4.9214,
                                 -4.0040, -4.1752, -3.9635, -3.4715, -4.3203, -4.0752, -4.9912, -4.1698,
                                 -3.2736, -3.1498, -4.2077, -4.0465, -3.1281, -4.4350, -3.3681, -3.8296,
                                 -4.4867, -4.9735, -4.7579, -4.5638, -3.6463, -3.4427, -4.3195, -3.4980,
                                 -3.7562, -4.3910, -3.6503, -4.3534, -3.7615, -4.7686, -4.6419, -3.4613,
                                 -3.6195, -3.7531, -4.5105, -4.4046, -4.9602, -4.6309, -3.0985, -3.1944,
                                 -4.5577, -3.1597, -4.4176, -4.1528, -4.9887, -4.9503, -3.5599, -3.3988,
                                 -3.2436, -4.2304, -3.7956, -4.1976, -3.0575, -3.6660, -4.2537, -4.5591,
                                 -3.2229, -4.4270, -4.8612, -4.0269, -4.8755, -3.9046, -3.8673, -3.4468,
                                 -4.3002, -3.2876, -4.6104, -3.6005, -4.8322, -3.5000, -3.4247, -3.8420,
                                 -4.1549, -4.7709, -3.8369, -3.4881, -4.6610, -3.1323, -3.4659, -3.0438,
                                 -4.5271, -4.2025, -4.6220, -4.6463, -3.6272, -3.2863, -4.6085, -4.2816,
                                 -4.1867, -3.1684, -3.5027, -3.6167, -3.7461, -3.2241, -4.7670, -3.1925,
                                 -3.2066, -4.9076, -4.5946, -3.9007, -3.6993, -4.0773, -3.0972, -3.0878,
                                 -3.1604, -3.7623, -3.8059, -4.4353, -3.7705, -3.3951, -4.1977, -3.5260,
                                 -3.4707, -4.8865, -4.2135, -4.0055, -4.0296, -3.1264, -3.0969, -3.9033,
                                 -3.3734, -3.4733, -3.7550, -3.5142, -3.2393, -4.9408, -3.0929, -3.6023,
                                 -4.4253, -4.4549, -4.4564, -4.8225, -4.9845, -3.1615, -3.5956, -4.0742,
                                 -3.0906, -4.7878, -4.5033, -4.3787, -4.0411, -4.6421, -4.2037, -4.9878,
                                 -4.9259, -4.8484, -3.1543, -3.3679, -3.1335, -3.0947, -4.9025, -3.5940,
                                 -4.3865, -4.1579, -3.9583, -3.4273, -4.8356, -3.3930, -3.7668, -3.1469,
                                 -4.3154, -3.7607, -3.6101, -3.0452, -4.7854, -3.6627, -4.5453, -3.8601,
                                 -4.8748, -3.0521, -3.4868, -3.0956, -3.9758, -3.8259, -4.0992, -3.9097,
                                 -3.6892, -4.6453, -4.1959, -3.8752, -3.7602, -4.6163, -4.6874, -3.1859,
                                 -4.5932, -3.0422, -4.8670, -3.1599, -4.1793, -3.5263, -4.0736, -4.9045,
                                 -4.2127, -3.4067, -4.3657, -3.3203, -4.9148, -4.1255, -3.9527, -4.7590,
                                 -3.7280, -4.9736, -4.0352, -3.4094, -3.0518, -4.3819, -3.4116, -4.1716,
                                 -3.9342, -3.3656, -4.5115, -3.2778, -4.0540, -4.9972, -3.8604, -3.7627,
                                 -3.7632, -4.8537, -3.5418, -3.4223, -4.0449, -3.1544, -4.7107, -3.2911,
                                 -3.6976, -3.9260, -4.4403, -3.5231, -4.6556, -3.3999, -3.3508, -3.9013,
                                 -3.0027, -4.9337, -4.9069, -3.7977, -4.6417, -4.8770, -3.0654, -3.8124,
                                 -3.1560, -4.5376, -3.8177, -3.6534, -3.2270, -3.1835, -3.2765, -3.2991,
                                 -3.2890, -3.1797, -3.0790, -4.4270, -3.1518, -4.9093, -3.1856, -3.5224,
                                 -3.5617, -3.4387, -4.7737, -4.4822, -4.7242, -3.6829, -3.7939, -3.1969,
                                 -4.4129, -4.6178, -4.9948, -3.6381, -4.4798, -4.4851, -3.1571, -4.6075,
                                 -4.5502, -4.0915, -3.7304, -3.4718, -4.8741, -4.4001, -4.6377, -3.9685,
                                 -4.3689, -4.4676, -4.2602, -4.7121, -3.1523, -3.6424, -3.6009, -4.4318,
                                 -3.0989, -4.3726, -4.5014, -4.5105, -4.2596, -4.4394, -3.2324, -3.2727,
                                 -4.6935, -4.1582, -4.3918, -4.1450, -4.6392, -3.0026, -3.5638, -4.8581,
                                 -4.7532, -4.6696, -3.4292, -3.0400, -4.1359, -4.9939, -4.6924, -3.1443,
                                 -4.0049, -4.0268, -3.9902, -3.3830, -4.9571, -4.7229, -4.3528, -4.6350,
                                 -3.4691, -4.9914, -4.1311, -3.2291, -3.7132, -3.5413, -4.8257, -3.3127,
                                 -3.0734, -3.6736, -3.0722, -3.0333, -3.4048, -4.2695, -3.3244, -4.2191,
                                 -4.0724, -4.1942, -3.1760, -3.1798, -4.7968, -3.9727, -3.9821, -4.1759,
                                 -4.4787, -4.3380, -3.7629, -3.2746, -3.4095, -4.4460, -3.4180, -3.6826,
                                 -4.1277, -4.7355, -3.5826, -4.9403, -4.7097, -3.6677, -4.8778, -3.5176,
                                 -3.7914, -3.1520, -4.5977, -3.2071, -3.2338, -4.5161, -4.1380, -3.1663,
                                 -3.2396, -3.0880, -3.5316, -3.9037, -4.1530, -3.2629, -4.9414, -4.4714,
                                 -4.4692, -4.0722, -3.9157, -3.3415, -4.3644, -3.3861, -3.4271, -4.3379,
                                 -4.4838, -3.1129, -4.0376, -4.5295, -3.6828, -3.5031, -4.7463, -4.0706,
                                 -3.2682, -3.6123, -3.6271, -3.1890, -3.5005, -4.2843, -3.9583, -3.2941,
                                 -3.1335, -3.4416, -4.6009, -3.7848, -4.3821, -3.2820, -4.8941, -3.7853,
                                 -4.9022, -4.5050, -4.1529, -4.7280, -3.7280, -4.3041, -4.8376, -3.9816,
                                 -3.1082, -3.0940, -4.7815, -3.4036, -4.7536, -3.7658, -4.7677, -3.3771,
                                 -3.2309, -3.4814, -3.5246, -3.6710, -4.9589, -3.1239, -3.3789, -4.3206])
    input_targets = torch.tensor([False, True, False, False, False, True, False, True, True, False,
                                  True, False, True, False, True, False, False, True, True, False,
                                  False, True, True, True, False, False, True, False, True, True,
                                  False, True, False, False, True, True, False, False, True, False,
                                  False, False, False, True, False, False, True, False, True, False,
                                  False, True, True, False, False, True, True, False, True, False,
                                  False, False, False, True, False, False, False, True, True, False,
                                  False, True, True, True, False, False, True, True, False, False,
                                  False, False, False, False, False, False, False, False, False, True,
                                  False, True, False, False, False, False, False, False, False, False,
                                  True, False, False, False, False, True, True, False, False, True,
                                  True, True, False, True, True, False, False, False, True, False,
                                  True, True, False, False, True, True, True, False, False, True,
                                  True, False, False, True, True, True, False, False, False, True,
                                  False, True, True, False, True, True, False, False, False, False,
                                  False, False, False, False, True, False, False, True, False, False,
                                  False, True, False, False, False, False, True, True, False, False,
                                  True, True, False, False, True, False, False, True, True, True,
                                  False, True, True, False, False, False, False, False, True, False,
                                  True, True, False, True, True, False, False, False, False, False,
                                  True, False, True, False, False, False, False, False, False, False,
                                  True, False, True, True, False, False, True, True, True, False,
                                  True, True, False, False, True, False, False, False, False, True,
                                  False, False, True, True, True, True, True, False, False, False,
                                  False, False, True, True, False, False, True, True, False, True,
                                  True, False, False, True, False, False, False, True, False, False,
                                  False, False, False, True, False, False, True, True, False, False,
                                  False, True, True, False, False, True, False, False, True, False,
                                  True, True, True, True, False, True, False, False, False, True,
                                  True, True, True, False, True, True, False, True, False, False,
                                  False, True, False, False, False, True, False, False, False, False,
                                  False, False, False, False, True, False, True, False, True, False,
                                  False, True, True, False, True, True, False, False, True, False,
                                  True, True, False, True, False, False, False, False, False, True,
                                  True, True, False, False, False, False, False, False, False, True,
                                  True, False, False, True, False, True, True, False, False, False,
                                  False, True, False, False, True, False, False, False, True, False,
                                  True, False, False, True, False, True, True, True, True, True,
                                  False, True, True, False, True, True, False, False, False, False,
                                  False, True, True, False, False, False, False, False, False, True,
                                  False, False, False, False, False, False, False, True, True, False,
                                  False, False, False, True, False, True, False, False, False, False,
                                  False, False, True, False, False, False, True, True, False, True,
                                  False, True, True, False, False, False, True, False, False, True,
                                  False, False, True, True, True, False, True, False, False, True,
                                  False, False, False, False, True, False, True, False, False, False,
                                  False, True, False, True, False, False, False, True, True, True,
                                  True, False, False, False, False, True, True, True, False, True,
                                  False, False, True, True, False, False, False, True, False, True,
                                  False, True, True, True, True, True, True, False, False, True,
                                  False, False, True, False, False, False, True, False, False, False,
                                  False, True, True, False, False, False, False, False, False, False,
                                  False, False, False, True, False, False, True, True, True, False,
                                  True, True, False, False, False, True, True, False, True, True,
                                  False, True, False, False, False, False, True, False, False, True,
                                  False, True, False, False, True, False, True, True, False, False,
                                  True, False, False, True, False, True, False, True, False, False,
                                  True, False, True, False, False, False, True, False, False, False,
                                  False, False, True, True, True, True, True, True, False, False,
                                  True, True, False, False, False, False, False, False, False, False,
                                  True, False, True, False, False, False, False, True, True, False,
                                  False, False, False, True, False, False, False, True, False, False,
                                  False, False, False, False, False, False, False, False, True, True,
                                  False, False, False, False, False, False, True, True, False, True,
                                  True, False, False, False, False, True, False, False, False, False,
                                  True, False, False, True, True, True, True, False, True, False,
                                  False, True, False, False, False, False, True, True, True, True,
                                  False, False, True, True, True, True, False, False, False, False,
                                  False, False, True, False, False, True, False, True, False, True,
                                  False, True, False, False, False, True, False, False, False, True,
                                  False, False, True, False, False, True, False, False, True, False,
                                  False, True, False, False, False, True, True, True, True, False,
                                  False, False, False, True, False, True, False, True, False, False,
                                  True, True, False, True, True, False, True, True, True, False,
                                  True, False, False, True, True, True, False, False, True, True,
                                  True, False, False, True, False, False, False, True, False, True,
                                  True, False, False, False, False, True, False, True, False, False,
                                  False, False, False, True, False, True, False, True, False, False,
                                  True, True, False, True, True, True, False, False, False, False,
                                  True, True, True, False, False, False, False, False, False, True,
                                  True, False, True, False, True, False, False, False, True, False,
                                  True, False, False, False, False, False, False, True, True, False,
                                  False, False, False, False, True, False, False, True, True, False,
                                  False, False, False, False, False, False, True, True, True, True,
                                  False, True, True, False, False, False, True, False, False, False,
                                  True, False, False, True, False, False, False, False, True, True,
                                  True, True, False, False, False, True, True, True, True, False,
                                  False, True, False, False, False, False, True, False, False, False,
                                  True, False, False, False, False, True, False, True, False, False,
                                  True, False, False, True, False, True, True, False, False, False,
                                  False, True, False, False, False, True, False, True, True, True,
                                  True, True, False, False, False, True, False, False, True, False,
                                  True, False, False, False, True, False, False, True, False, False,
                                  False, True, False, True, False, False, False, True, False, True,
                                  True, False, False, False, False, False, False, True, True, False,
                                  True, False, False, False, True, False, True, True, False, True,
                                  True, True, False, False, False, True, False, False, True, False,
                                  True, True, False, False, True, False, False, False, False, True,
                                  False, False, False, True, False, False, True, True, True, True,
                                  False, True, True, True, False, False, False, False, False, False])
    input_activation_function = torch.relu
    input_threshold = torch.tensor(0.0)
    print(ex6(input_logits, input_activation_function, input_threshold, input_targets))
    print(input_activation_function(input_logits) > input_threshold)
