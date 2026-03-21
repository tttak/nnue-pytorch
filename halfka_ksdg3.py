import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

class Features(FeatureBlock):
  def __init__(self):
    super(Features, self).__init__('HalfKA_KSDG3', 0x5f134cb8 ^ (0x596B2374 << 1) ^ (0x596B2374 >> 31), OrderedDict([('HalfKA_KSDG3', 190998 + 12672)]))

class FactorizedFeatures(FeatureBlock):
  def __init__(self):
    super(FactorizedFeatures, self).__init__('HalfKA_KSDG3^', 0x5f134cb8 ^ (0x596B2374 << 1) ^ (0x596B2374 >> 31), OrderedDict([('HalfKA_KSDG3', 190998 + 12672), ('A_GOLDS', 2358), ('HalfRelKAGOLDS', 8182), ('KSDGE00_GOLDS', 12672)]))
    self.base = Features()

  def _make_relka_index(self, sq_k, p):
    if p < 90:
      return p
    w = 9 * 2 - 1
    h = 9 * 2 - 1
    piece_index = (p - 90) // 81
    sq_p = (p - 90) % 81
    relative_file = (sq_p // 9) - (sq_k // 9) + (w // 2)
    relative_rank = (sq_p % 9) - (sq_k % 9) + (h // 2)
    return int(h * w * piece_index + h * relative_file + relative_rank + 90)

  def get_feature_factors(self, idx):
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    # KSDG3の場合
    if idx < 12672:
      idx_tmp = idx

      # インデックスの各要素を逆算する
      idx_effect2 = idx_tmp %  4
      idx_tmp     = idx_tmp // 4

      idx_effect1 = idx_tmp %  4
      idx_tmp     = idx_tmp // 4

      idx_pc      = idx_tmp %  33
      idx_tmp     = idx_tmp // 33

      idx_dir     = idx_tmp

      # 「と金～成銀」を金と同一視する
      if 9 <= idx_pc <= 12:
        idx_pc = 7
      if 25 <= idx_pc <= 28:
        idx_pc = 23

      # 利き数の相違を無視する
      idx_effect1 = 0
      idx_effect2 = 0

      idx_KSDGE00_GOLDS = ((idx_dir * 33 + idx_pc) * 4 + idx_effect1) * 4 + idx_effect2

      return [idx, self.get_factor_base_feature('KSDGE00_GOLDS') + idx_KSDGE00_GOLDS]

    # HalfKA_DGの場合
    else:
      idx_HalfKA_DG = idx - 12672

      idx_A = idx_HalfKA_DG %  2358
      idx_K = idx_HalfKA_DG // 2358

      # 「と金～成銀」を金と同一視する
      if 1548 <= idx_A < 2196:
        idx_A_GOLDS = (idx_A - 1548) % 162 + 738
      else:
        idx_A_GOLDS = idx_A

      return [idx, self.get_factor_base_feature('A_GOLDS') + idx_A_GOLDS, self.get_factor_base_feature('HalfRelKAGOLDS') + self._make_relka_index(idx_K, idx_A_GOLDS)]

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures]
