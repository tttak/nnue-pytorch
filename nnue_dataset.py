import numpy as np
import ctypes
import torch
import os
import sys
import glob
from torch.utils.data import Dataset

loader_override = os.environ.get('NNUE_TRAINING_DATA_LOADER')
local_dllpath = ([loader_override] if loader_override else
                 [n for n in glob.glob('./*training_data_loader.*')
                  if n.endswith('.so') or n.endswith('.dll')
                  or n.endswith('.dylib')])
if not local_dllpath:
    print('Cannot find data_loader shared library.')
    sys.exit(1)
dllpath = os.path.abspath(local_dllpath[0])
dll = ctypes.cdll.LoadLibrary(dllpath)

class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('size', ctypes.c_int),
        ('is_white', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('ranking_target', ctypes.POINTER(ctypes.c_float)),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('max_active_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_int)),
        ('black', ctypes.POINTER(ctypes.c_int)),
        ('white_values', ctypes.POINTER(ctypes.c_float)),
        ('black_values', ctypes.POINTER(ctypes.c_float)),
        ('layer_stack_indices', ctypes.POINTER(ctypes.c_int)),
        ('material', ctypes.POINTER(ctypes.c_float)),
        ('kif_group_id', ctypes.POINTER(ctypes.c_int)),
        ('ply', ctypes.POINTER(ctypes.c_int)),
    ]

    def get_tensors(self, device, include_ranking_target=False,
                    side_input="none", pair_relation_side_input=False):
        white_values = torch.from_numpy(np.ctypeslib.as_array(self.white_values, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        black_values = torch.from_numpy(np.ctypeslib.as_array(self.black_values, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        white_indices = torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        black_indices = torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.size, self.max_active_features))).pin_memory().to(device=device, non_blocking=True)
        us = torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        them = 1.0 - us
        outcome = torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        score = torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        ranking_target = torch.from_numpy(np.ctypeslib.as_array(
            self.ranking_target, shape=(self.size, 1))).pin_memory().to(
                device=device, non_blocking=True)
        layer_stack_indices = torch.from_numpy(np.ctypeslib.as_array(self.layer_stack_indices, shape=(self.size,))).long().pin_memory().to(device=device, non_blocking=True)
        material = torch.from_numpy(np.ctypeslib.as_array(self.material, shape=(self.size, 1))).pin_memory().to(device=device, non_blocking=True)
        kif_group_id = torch.from_numpy(np.ctypeslib.as_array(self.kif_group_id, shape=(self.size,))).long().pin_memory().to(device=device, non_blocking=True)
        ply = torch.from_numpy(np.ctypeslib.as_array(self.ply, shape=(self.size,))).long().pin_memory().to(device=device, non_blocking=True)
        result = (us, them, white_indices, white_values, black_indices,
                  black_values, outcome, score, layer_stack_indices, material,
                  kif_group_id, ply)
        if include_ranking_target:
            result += (ranking_target,)
        if side_input == "safe_escape":
            if get_sparse_batch_safe_escape is None:
                raise RuntimeError(
                    "training_data_loader lacks safe-escape side-input ABI; "
                    "rebuild training_data_loader.dll")
            pointer = get_sparse_batch_safe_escape(ctypes.byref(self))
            packed = np.ctypeslib.as_array(pointer, shape=(self.size,)).copy()
            bits = ((packed[:, None] >> np.arange(16, dtype=np.uint16)) & 1)
            side = torch.from_numpy(bits.astype(np.float32)).pin_memory().to(
                device=device, non_blocking=True)
            result += (side,)
        elif side_input in ("mobility_tactical_v1", "mobility_tactical_v2"):
            if get_sparse_batch_mobility_tactical is None:
                raise RuntimeError(
                    "training_data_loader lacks mobility/tactical side-input ABI; "
                    "rebuild training_data_loader.dll")
            pointer = get_sparse_batch_mobility_tactical(ctypes.byref(self))
            values = np.ctypeslib.as_array(
                pointer, shape=(self.size, 8)).copy()
            side = torch.from_numpy(values).pin_memory().to(
                device=device, non_blocking=True)
            result += (side,)
        if pair_relation_side_input:
            if (get_sparse_batch_pair_relation_count is None
                    or get_sparse_batch_pair_relation_indices is None
                    or get_sparse_batch_pair_relation_batch_indices is None):
                raise RuntimeError(
                    "training_data_loader lacks pair-relation v1 ABI; "
                    "rebuild training_data_loader.dll")
            count = int(get_sparse_batch_pair_relation_count(
                ctypes.byref(self)))
            if count:
                indices_pointer = get_sparse_batch_pair_relation_indices(
                    ctypes.byref(self))
                batch_pointer = get_sparse_batch_pair_relation_batch_indices(
                    ctypes.byref(self))
                indices_np = np.ctypeslib.as_array(
                    indices_pointer, shape=(count,)).copy()
                batch_np = np.ctypeslib.as_array(
                    batch_pointer, shape=(count,)).copy()
            else:
                indices_np = np.empty((0,), dtype=np.int32)
                batch_np = np.empty((0,), dtype=np.int32)
            pair_indices = torch.from_numpy(indices_np).long().pin_memory().to(
                device=device, non_blocking=True)
            pair_batch_indices = torch.from_numpy(batch_np).long().pin_memory().to(
                device=device, non_blocking=True)
            result += (pair_indices, pair_batch_indices)
        return result


SparseBatchPtr = ctypes.POINTER(SparseBatch)

class TrainingDataProvider:
    def __init__(
        self,
        feature_set,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filename1,
        filename2,
        filename3,
        train1_rate,
        train2_rate,
        skiprate,
        mirror,
        cyclic,
        num_workers,
        batch_size=None,
        filtered=False,
        random_fen_skipping=0,
        device='cpu',
        ranking_target3=None, side_input="none",
        pair_relation_side_input=False):

        self.feature_set = feature_set.encode('utf-8')
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filename1 = filename1.encode('utf-8')
        self.filename2 = filename2.encode('utf-8')
        self.filename3 = filename3.encode('utf-8')
        self.train1_rate = train1_rate
        self.train2_rate = train2_rate
        self.skiprate = skiprate
        self.mirror = mirror
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.filtered = filtered
        self.random_fen_skipping = random_fen_skipping
        self.device = device
        self.ranking_target3 = ranking_target3
        self.side_input = side_input
        self.pair_relation_side_input = bool(pair_relation_side_input)

        if batch_size:
            if ranking_target3:
                if self.pair_relation_side_input:
                    create = create_sparse_batch_stream_with_ranking_target3_pair_relation
                elif self.side_input in ("mobility_tactical_v1", "mobility_tactical_v2"):
                    create = create_sparse_batch_stream_with_ranking_target3_mobility_tactical
                else:
                    create = create_sparse_batch_stream_with_ranking_target3
                if create is None:
                    raise RuntimeError(
                        "training_data_loader lacks pair-relation v1 stream ABI; "
                        "rebuild training_data_loader.dll")
                self.stream = create(
                    self.feature_set, self.num_workers, self.filename1,
                    self.filename2, self.filename3,
                    os.fsencode(ranking_target3), train1_rate, train2_rate,
                    skiprate, mirror, batch_size, cyclic, filtered,
                    random_fen_skipping)
            else:
                if self.pair_relation_side_input:
                    create = create_sparse_batch_stream_pair_relation
                elif self.side_input in ("mobility_tactical_v1", "mobility_tactical_v2"):
                    create = create_sparse_batch_stream_mobility_tactical
                else:
                    create = self.create_stream
                if create is None:
                    raise RuntimeError(
                        "training_data_loader lacks pair-relation v1 stream ABI; "
                        "rebuild training_data_loader.dll")
                self.stream = create(self.feature_set, self.num_workers, self.filename1, self.filename2, self.filename3, train1_rate, train2_rate, skiprate, mirror, batch_size, cyclic, filtered, random_fen_skipping)
        else:
            self.stream = self.create_stream(self.feature_set, self.num_workers, self.filename1, self.filename2, self.filename3, train1_rate, train2_rate, skiprate, mirror, cyclic, filtered, random_fen_skipping)

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)

        if v:
            tensors = v.contents.get_tensors(
                self.device, include_ranking_target=bool(self.ranking_target3),
                side_input=self.side_input,
                pair_relation_side_input=self.pair_relation_side_input)
            self.destroy_part(v)
            return tensors
        else:
            raise StopIteration

    def __del__(self):
        self.destroy_stream(self.stream)

create_sparse_batch_stream = dll.create_sparse_batch_stream
create_sparse_batch_stream.restype = ctypes.c_void_p
create_sparse_batch_stream.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

create_sparse_batch_stream_with_ranking_target3 = (
    dll.create_sparse_batch_stream_with_ranking_target3)
create_sparse_batch_stream_with_ranking_target3.restype = ctypes.c_void_p
create_sparse_batch_stream_with_ranking_target3.argtypes = [
    ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p,
    ctypes.c_char_p, ctypes.c_char_p, ctypes.c_float, ctypes.c_float,
    ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_int]
destroy_sparse_batch_stream = dll.destroy_sparse_batch_stream
destroy_sparse_batch_stream.argtypes = [ctypes.c_void_p]

fetch_next_sparse_batch = dll.fetch_next_sparse_batch
fetch_next_sparse_batch.restype = SparseBatchPtr
fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]
destroy_sparse_batch = dll.destroy_sparse_batch

try:
    get_sparse_batch_safe_escape = dll.get_sparse_batch_safe_escape
    get_sparse_batch_safe_escape.restype = ctypes.POINTER(ctypes.c_uint16)
    get_sparse_batch_safe_escape.argtypes = [SparseBatchPtr]
except AttributeError:
    get_sparse_batch_safe_escape = None

try:
    create_sparse_batch_stream_mobility_tactical = (
        dll.create_sparse_batch_stream_mobility_tactical)
    create_sparse_batch_stream_mobility_tactical.restype = ctypes.c_void_p
    create_sparse_batch_stream_mobility_tactical.argtypes = (
        create_sparse_batch_stream.argtypes)
    create_sparse_batch_stream_with_ranking_target3_mobility_tactical = (
        dll.create_sparse_batch_stream_with_ranking_target3_mobility_tactical)
    create_sparse_batch_stream_with_ranking_target3_mobility_tactical.restype = (
        ctypes.c_void_p)
    create_sparse_batch_stream_with_ranking_target3_mobility_tactical.argtypes = (
        create_sparse_batch_stream_with_ranking_target3.argtypes)
    get_sparse_batch_mobility_tactical = (
        dll.get_sparse_batch_mobility_tactical)
    get_sparse_batch_mobility_tactical.restype = ctypes.POINTER(ctypes.c_float)
    get_sparse_batch_mobility_tactical.argtypes = [SparseBatchPtr]
except AttributeError:
    create_sparse_batch_stream_mobility_tactical = None
    create_sparse_batch_stream_with_ranking_target3_mobility_tactical = None
    get_sparse_batch_mobility_tactical = None

try:
    create_sparse_batch_stream_pair_relation = (
        dll.create_sparse_batch_stream_pair_relation)
    create_sparse_batch_stream_pair_relation.restype = ctypes.c_void_p
    create_sparse_batch_stream_pair_relation.argtypes = (
        create_sparse_batch_stream.argtypes)
    create_sparse_batch_stream_with_ranking_target3_pair_relation = (
        dll.create_sparse_batch_stream_with_ranking_target3_pair_relation)
    create_sparse_batch_stream_with_ranking_target3_pair_relation.restype = (
        ctypes.c_void_p)
    create_sparse_batch_stream_with_ranking_target3_pair_relation.argtypes = (
        create_sparse_batch_stream_with_ranking_target3.argtypes)
    get_sparse_batch_pair_relation_count = (
        dll.get_sparse_batch_pair_relation_count)
    get_sparse_batch_pair_relation_count.restype = ctypes.c_size_t
    get_sparse_batch_pair_relation_count.argtypes = [SparseBatchPtr]
    get_sparse_batch_pair_relation_indices = (
        dll.get_sparse_batch_pair_relation_indices)
    get_sparse_batch_pair_relation_indices.restype = (
        ctypes.POINTER(ctypes.c_int32))
    get_sparse_batch_pair_relation_indices.argtypes = [SparseBatchPtr]
    get_sparse_batch_pair_relation_batch_indices = (
        dll.get_sparse_batch_pair_relation_batch_indices)
    get_sparse_batch_pair_relation_batch_indices.restype = (
        ctypes.POINTER(ctypes.c_int32))
    get_sparse_batch_pair_relation_batch_indices.argtypes = [SparseBatchPtr]
    get_sparse_batch_source_sfen = dll.get_sparse_batch_source_sfen
    get_sparse_batch_source_sfen.restype = ctypes.c_char_p
    get_sparse_batch_source_sfen.argtypes = [SparseBatchPtr, ctypes.c_size_t]
except AttributeError:
    create_sparse_batch_stream_pair_relation = None
    create_sparse_batch_stream_with_ranking_target3_pair_relation = None
    get_sparse_batch_pair_relation_count = None
    get_sparse_batch_pair_relation_indices = None
    get_sparse_batch_pair_relation_batch_indices = None
    get_sparse_batch_source_sfen = None

get_sparse_batch_from_fens = dll.get_sparse_batch_from_fens
get_sparse_batch_from_fens.restype = SparseBatchPtr
get_sparse_batch_from_fens.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
try:
    get_sparse_batch_from_fens_pair_relation = (
        dll.get_sparse_batch_from_fens_pair_relation)
    get_sparse_batch_from_fens_pair_relation.restype = SparseBatchPtr
    get_sparse_batch_from_fens_pair_relation.argtypes = (
        get_sparse_batch_from_fens.argtypes)
except AttributeError:
    get_sparse_batch_from_fens_pair_relation = None

def make_sparse_batch_from_fens(feature_set, fens, scores, plies, results,
                                pair_relation_side_input=False):
    results_ = (ctypes.c_int*len(scores))()
    scores_ = (ctypes.c_int*len(plies))()
    plies_ = (ctypes.c_int*len(results))()
    fens_ = (ctypes.c_char_p * len(fens))()
    fens_[:] = [fen.encode('utf-8') for fen in fens]
    for i, v in enumerate(scores):
        scores_[i] = v
    for i, v in enumerate(plies):
        plies_[i] = v
    for i, v in enumerate(results):
        results_[i] = v
    create = (get_sparse_batch_from_fens_pair_relation
              if pair_relation_side_input else get_sparse_batch_from_fens)
    if create is None:
        raise RuntimeError("training_data_loader lacks pair-relation v1 ABI")
    b = create(feature_set.name.encode('utf-8'), len(fens), fens_, scores_, plies_, results_)
    return b

class SparseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename1, filename2, filename3, train1_rate, train2_rate, skiprate, mirror, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, device='cpu', ranking_target3=None, side_input="none", pair_relation_side_input=False):
        super(SparseBatchProvider, self).__init__(
            feature_set,
            create_sparse_batch_stream,
            destroy_sparse_batch_stream,
            fetch_next_sparse_batch,
            destroy_sparse_batch,
            filename1,
            filename2,
            filename3,
            train1_rate,
            train2_rate,
            skiprate,
            mirror,
            cyclic,
            num_workers,
            batch_size,
            filtered,
            random_fen_skipping,
            device,
            ranking_target3, side_input, pair_relation_side_input)

class SparseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filename1, filename2, filename3, train1_rate, train2_rate, skiprate, mirror, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, device='cpu', ranking_target3=None, side_input="none", pair_relation_side_input=False):
    super(SparseBatchDataset).__init__()
    self.feature_set = feature_set
    self.filename1 = filename1
    self.filename2 = filename2
    self.filename3 = filename3
    self.train1_rate = train1_rate
    self.train2_rate = train2_rate
    self.skiprate = skiprate
    self.mirror = mirror
    self.batch_size = batch_size
    self.cyclic = cyclic
    self.num_workers = num_workers
    self.filtered = filtered
    self.random_fen_skipping = random_fen_skipping
    self.device = device
    self.ranking_target3 = ranking_target3
    self.side_input = side_input
    self.pair_relation_side_input = bool(pair_relation_side_input)

  def __iter__(self):
    return SparseBatchProvider(self.feature_set, self.filename1, self.filename2, self.filename3, self.train1_rate, self.train2_rate, self.skiprate, self.mirror, self.batch_size, cyclic=self.cyclic, num_workers=self.num_workers, filtered=self.filtered, random_fen_skipping=self.random_fen_skipping, device=self.device, ranking_target3=self.ranking_target3, side_input=self.side_input, pair_relation_side_input=self.pair_relation_side_input)

class FixedNumBatchesDataset(Dataset):
  def __init__(self, dataset, num_batches):
    super(FixedNumBatchesDataset, self).__init__()
    self.dataset = dataset;
    self.iter = iter(self.dataset)
    self.num_batches = num_batches

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    return next(self.iter)
