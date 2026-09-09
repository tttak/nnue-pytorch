import argparse
import atexit
import builtins
from datetime import datetime
import model as M
import nnue_dataset
import nnue_bin_dataset
import pytorch_lightning as pl
import features
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import torch
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning.callbacks


class GracefulInterruptController:
  """Turn the first SIGINT into KeyboardInterrupt and force-exit on repeats."""

  def __init__(self):
    self.graceful_interrupt_in_progress = False
    self._previous_sigint_handler = None
    self._previous_sigbreak_handler = None
    self._force_exit_watchdog = None

  def install(self) -> None:
    self._previous_sigint_handler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, self._handle_sigint)
    if os.name == 'nt':
      self._previous_sigbreak_handler = signal.getsignal(signal.SIGBREAK)
      signal.signal(signal.SIGBREAK, self._handle_sigint)

  def restore(self) -> None:
    if self._previous_sigint_handler is not None:
      signal.signal(signal.SIGINT, self._previous_sigint_handler)
      self._previous_sigint_handler = None
    if self._previous_sigbreak_handler is not None:
      signal.signal(signal.SIGBREAK, self._previous_sigbreak_handler)
      self._previous_sigbreak_handler = None

  def begin_graceful_interrupt(self) -> None:
    self.graceful_interrupt_in_progress = True

  def start_force_exit_watchdog(self) -> None:
    """Let a second Ctrl+C kill us even while checkpoint code holds the GIL."""
    if self._force_exit_watchdog is not None:
      return

    # This separate process shares the console but not the checkpoint writer's
    # GIL.  It starts only after the first Ctrl+C, so the next SIGINT it sees is
    # necessarily the user's second Ctrl+C.
    watchdog_code = r'''
import ctypes
import os
import signal
import sys
import time

parent_pid = int(sys.argv[1])
parent_process = None
if os.name == 'nt':
  parent_process = ctypes.windll.kernel32.OpenProcess(
      0x0001 | 0x00100000, False, parent_pid)  # TERMINATE | SYNCHRONIZE
  if not parent_process:
    raise OSError('Cannot open parent process for Ctrl+C watchdog')

def force_exit(signum, frame):
  try:
    os.write(2, b'Second Ctrl+C detected. Exiting immediately without saving checkpoint.\n')
    if os.name == 'nt':
      ctypes.windll.kernel32.TerminateProcess(parent_process, 130)
    else:
      os.kill(parent_pid, signal.SIGKILL)
  finally:
    os._exit(130)

signal.signal(signal.SIGINT, force_exit)
if os.name == 'nt':
  signal.signal(signal.SIGBREAK, force_exit)
print('READY', flush=True)
while True:
  if os.name == 'nt':
    if ctypes.windll.kernel32.WaitForSingleObject(parent_process, 1000) == 0:
      ctypes.windll.kernel32.CloseHandle(parent_process)
      os._exit(0)
  else:
    try:
      os.kill(parent_pid, 0)
    except ProcessLookupError:
      os._exit(0)
    time.sleep(1)
'''
    self._force_exit_watchdog = subprocess.Popen(
        [sys.executable, '-c', watchdog_code, str(os.getpid())],
        stdout=subprocess.PIPE,
        stderr=None,
        text=True)
    ready = self._force_exit_watchdog.stdout.readline().strip()
    if ready != 'READY':
      self.stop_force_exit_watchdog()
      raise RuntimeError('Failed to start the second-Ctrl+C watchdog')

  def stop_force_exit_watchdog(self) -> None:
    watchdog = self._force_exit_watchdog
    self._force_exit_watchdog = None
    if watchdog is None:
      return
    if watchdog.poll() is None:
      watchdog.terminate()
      try:
        watchdog.wait(timeout=5)
      except subprocess.TimeoutExpired:
        watchdog.kill()
        watchdog.wait()
    if watchdog.stdout is not None:
      watchdog.stdout.close()

  def _handle_sigint(self, signum, frame) -> None:
    if self.graceful_interrupt_in_progress:
      # Avoid buffered print()/logging here: the second Ctrl+C must not wait on
      # a checkpoint writer or another lock held by the interrupted process.
      try:
        # During checkpointing the independent watchdog prints this message;
        # do not duplicate it if both handlers receive the same console event.
        if self._force_exit_watchdog is None:
          os.write(
              2,
              b'Second Ctrl+C detected. Exiting immediately without saving checkpoint.\n')
      finally:
        os._exit(130)

    # Record the first signal before raising, so another Ctrl+C can abort even
    # while Lightning is unwinding into Callback.on_exception().
    self.begin_graceful_interrupt()
    signal.default_int_handler(signum, frame)


class TextLogPrintTee:
  """Duplicate implicit-destination print() calls to a UTF-8 text log.

  Calls which explicitly provide ``file=`` retain normal print semantics and
  are deliberately not copied.  In particular, tqdm/Lightning progress output
  written to its terminal stream stays out of the text log.
  """

  def __init__(self, requested_path, started_at):
    self.requested_path = Path(requested_path).expanduser().resolve()
    self.started_at = started_at
    self.path = None
    # Capture diagnostics printed before TensorBoardLogger selects its version.
    # The buffer spills to a temporary file if checkpoint-loading diagnostics
    # become large, then is copied into the final line-buffered log.
    self._stream = tempfile.SpooledTemporaryFile(
        max_size=1024 * 1024, mode="w+", encoding="utf-8", newline="")
    self._original_print = builtins.print
    self._lock = threading.RLock()
    self._closed = False

    def tee_print(*args, **kwargs):
      # An explicit destination belongs to the caller.  Do not redirect or
      # duplicate it, even when the caller explicitly uses file=sys.stdout.
      if "file" in kwargs:
        return self._original_print(*args, **kwargs)

      with self._lock:
        self._original_print(*args, **kwargs)
        log_kwargs = dict(kwargs)
        log_kwargs["file"] = self._stream
        self._original_print(*args, **log_kwargs)

    self._tee_print = tee_print
    builtins.print = self._tee_print
    atexit.register(self.close)

  def bind_lightning_version(self, lightning_version):
    """Select an overwrite-safe final path and flush deferred output into it."""
    with self._lock:
      if self._closed:
        raise RuntimeError("cannot bind a closed text log")
      if self.path is not None:
        return self.path

      requested = self.requested_path
      requested.parent.mkdir(parents=True, exist_ok=True)
      timestamp = self.started_at.strftime("%Y%m%d_%H%M%S")
      version = str(lightning_version)
      base_name = f"{requested.stem}_v{version}_{timestamp}"

      serial = 0
      while True:
        collision_suffix = "" if serial == 0 else f"_{serial:02d}"
        candidate = requested.with_name(
            f"{base_name}{collision_suffix}{requested.suffix}")
        try:
          final_stream = open(
              candidate, "x", encoding="utf-8", buffering=1, newline="")
          break
        except FileExistsError:
          serial += 1

      deferred_stream = self._stream
      deferred_stream.flush()
      deferred_stream.seek(0)
      while True:
        block = deferred_stream.read(1024 * 1024)
        if not block:
          break
        final_stream.write(block)
      final_stream.flush()
      deferred_stream.close()

      self._stream = final_stream
      self.path = str(candidate)
      return self.path

  def close(self):
    with self._lock:
      if self._closed:
        return
      if builtins.print is self._tee_print:
        builtins.print = self._original_print
      self._stream.flush()
      self._stream.close()
      self._closed = True

def data_loader_cc(train_filename1, train_filename2, train_filename3, val_filename, feature_set, num_workers, batch_size, filtered, random_fen_skipping, main_device, epoch_size, train1_rate, train2_rate, skiprate, mirror):
  # Epoch and validation sizes are arbitrary
  val_size = 1000000
  features_name = feature_set.name
  train_infinite = nnue_dataset.SparseBatchDataset(features_name, train_filename1, train_filename2, train_filename3, train1_rate, train2_rate, skiprate, mirror, batch_size, num_workers=num_workers,
                                                   filtered=filtered, random_fen_skipping=random_fen_skipping, device=main_device)
  val_infinite = nnue_dataset.SparseBatchDataset(features_name, val_filename, val_filename, val_filename, train1_rate, train2_rate, skiprate, 0.00, batch_size, filtered=filtered,
                                                   random_fen_skipping=random_fen_skipping, device=main_device)
  # num_workers has to be 0 for sparse, and 1 for dense
  # it currently cannot work in parallel mode but it shouldn't need to
  train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  return train, val

def data_loader_py(train_filename, val_filename, feature_set, batch_size, main_device):
  train = DataLoader(nnue_bin_dataset.NNUEBinData(train_filename, feature_set), batch_size=batch_size, shuffle=True, num_workers=4)
  val = DataLoader(nnue_bin_dataset.NNUEBinData(val_filename, feature_set), batch_size=32)
  return train, val

class NetworkSaveCheckpoint(pytorch_lightning.callbacks.Checkpoint):
  def __init__(
      self,
      every_n_epochs: int,
      log_dir: str,
      interrupt_controller: GracefulInterruptController | None = None,
  ):
    self.every_n_epochs = every_n_epochs
    self.log_dir = log_dir
    self.interrupt_controller = interrupt_controller
    self.final_checkpoint_saved = False
    self.keyboard_interrupt_handled = False

  def on_validation_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') -> None:
    if self.every_n_epochs != 1 and (trainer.current_epoch == 0 or trainer.current_epoch % self.every_n_epochs != 0):
      return
    ckpt_file_path = os.path.join(self.log_dir, f'{trainer.current_epoch}.ckpt')
    trainer.save_checkpoint(ckpt_file_path)

  def save_final_checkpoint(
      self,
      trainer: 'pl.Trainer',
      interruptible: bool = False,
  ) -> str:
    """Atomically save final.ckpt once, for normal completion or interruption."""
    ckpt_file_path = os.path.join(self.log_dir, 'final.ckpt')
    if not self.final_checkpoint_saved:
      # Keep a previously completed final.ckpt intact until the replacement is
      # fully written.  A forced second Ctrl+C may leave this exact .tmp file,
      # but can never expose it as final.ckpt.
      temporary_path = ckpt_file_path + '.tmp'
      use_watchdog = interruptible and self.interrupt_controller is not None
      if use_watchdog:
        # Serialization can delay Python's own signal handler.  The watchdog
        # is a separate process in the same console and can terminate this
        # process immediately when the user presses Ctrl+C again.
        self.interrupt_controller.start_force_exit_watchdog()
      try:
        trainer.save_checkpoint(temporary_path)
      finally:
        if use_watchdog:
          self.interrupt_controller.stop_force_exit_watchdog()
      os.replace(temporary_path, ckpt_file_path)
      self.final_checkpoint_saved = True
    return ckpt_file_path

  def on_exception(
      self,
      trainer: 'pl.Trainer',
      pl_module: 'pl.LightningModule',
      exception: BaseException,
  ) -> None:
    # Lightning 2.x consumes KeyboardInterrupt inside Trainer.fit(), invokes
    # this official hook, tears down, and then raises SystemExit(1).  Save while
    # the Trainer is still intact.  Do not checkpoint arbitrary failures.
    if not isinstance(exception, KeyboardInterrupt):
      return
    if self.interrupt_controller is not None:
      self.interrupt_controller.begin_graceful_interrupt()
    ckpt_file_path = self.save_final_checkpoint(trainer, interruptible=True)
    # Treat Lightning's trailing SystemExit as a graceful interrupt only after
    # final.ckpt was written successfully.
    self.keyboard_interrupt_handled = True
    print(f'KeyboardInterrupt checkpoint saved: {ckpt_file_path}', flush=True)

def main():
  started_at = datetime.now()
  parser = argparse.ArgumentParser(description="Trains the network.")
  parser.add_argument("train1", help="Training data (.bin or .binpack)")
  parser.add_argument("train2", help="Training data (.bin or .binpack)")
  parser.add_argument("train3", help="Training data (.bin or .binpack)")
  parser.add_argument("val", help="Validation data (.bin or .binpack)")
  # Lightning 2.x no longer exposes Trainer.add_argparse_args().  Keep the
  # command-line options used by the existing training scripts and translate
  # them to the current Trainer API below.
  parser.add_argument("--gpus", default=0, type=int,
                      help="Number of CUDA devices (0 selects CPU).")
  parser.add_argument("--max_epochs", "--max-epochs", default=None, type=int,
                      dest="max_epochs", help="Maximum number of training epochs.")
  parser.add_argument("--default_root_dir", "--default-root-dir", default=None,
                      dest="default_root_dir", help="Lightning root directory.")
  parser.add_argument("--log_every_n_steps", "--log-every-n-steps", default=50,
                      type=int, dest="log_every_n_steps",
                      help="How often Lightning logs training metrics.")
  parser.add_argument("--py-data", action="store_true", help="Use python data loader (default=False)")
  parser.add_argument("--lambda", default=1.0, type=float, dest='lambda_', help="lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0).")
  parser.add_argument("--start-lambda", default=None, type=float, dest='start_lambda', help="lambda to use at first epoch.")
  parser.add_argument("--end-lambda", default=None, type=float, dest='end_lambda', help="lambda to use at last epoch.")
  parser.add_argument("--gamma", default=0.992, type=float, dest='gamma', help="Multiplicative factor applied to the learning rate after every epoch.")
  parser.add_argument("--lr", default=8.75e-4, type=float, dest='lr', help="Initial learning rate.")
  parser.add_argument("--num-workers", default=1, type=int, dest='num_workers', help="Number of worker threads to use for data loading. Currently only works well for binpack.")
  parser.add_argument("--batch-size", default=-1, type=int, dest='batch_size', help="Number of positions per batch / per iteration. Default on GPU = 8192 on CPU = 128.")
  parser.add_argument("--threads", default=-1, type=int, dest='threads', help="Number of torch threads to use. Default automatic (cores) .")
  parser.add_argument("--seed", default=42, type=int, dest='seed', help="torch seed to use.")
  parser.add_argument("--smart-fen-skipping", action='store_true', dest='smart_fen_skipping', help="If enabled positions that are bad training targets will be skipped during loading. Default: False")
  parser.add_argument("--random-fen-skipping", default=0, type=int, dest='random_fen_skipping', help="skip fens randomly on average random_fen_skipping before using one.")
  parser.add_argument("--resume-from-model", dest='resume_from_model', help="Initializes training using the weights from the given .pt model")
  parser.add_argument("--epoch-size", default=1000000, type=int, dest='epoch_size', help="epoch size.")
  parser.add_argument("--in-scaling", default=240, type=int, dest='in_scaling', help="in-scaling.")
  parser.add_argument("--out-scaling", default=280, type=int, dest='out_scaling', help="out-scaling.")
  parser.add_argument("--offset", default=270, type=int, dest='offset', help="offset.")
  parser.add_argument("--offset1", default=270, type=int, dest='offset1', help="offset1.")
  parser.add_argument("--offset2", default=270, type=int, dest='offset2', help="offset2.")
  parser.add_argument("--adjust-loss", default=0.1, type=float, dest='adjust_loss', help="adjust loss.")
  parser.add_argument("--train1-rate", default=0.33, type=float, dest='train1_rate', help="train1-rate")
  parser.add_argument("--train2-rate", default=0.33, type=float, dest='train2_rate', help="train2-rate")
  parser.add_argument("--skiprate", default=1.5, type=float, dest='skiprate', help="skiprate")
  parser.add_argument("--mirror", default=0.00, type=float, dest='mirror', help="mirror")
  parser.add_argument("--network-save-period", type=int, default=1000000000, dest='network_save_period', help="Number of epochs between network snapshots. None to disable.")
  parser.add_argument("--text-log", dest="text_log", help="Duplicate ordinary print() output to this UTF-8 text file; progress bars remain terminal-only.")

  features.add_argparse_args(parser)
  args = parser.parse_args()

  text_log_tee = (
      TextLogPrintTee(args.text_log, started_at) if args.text_log else None)

  if not os.path.exists(args.train1):
    raise Exception('{0} does not exist'.format(args.train1))
  if not os.path.exists(args.train2):
    raise Exception('{0} does not exist'.format(args.train2))
  if not os.path.exists(args.train3):
    raise Exception('{0} does not exist'.format(args.train3))
  if not os.path.exists(args.val):
    raise Exception('{0} does not exist'.format(args.val))

  if not (0.0 < args.train1_rate < 1.0):
    raise ValueError(f"--train1-rate must be strictly between 0.0 and 1.0 (got {args.train1_rate})")
  if not (0.0 < args.train2_rate < 1.0):
    raise ValueError(f"--train2-rate must be strictly between 0.0 and 1.0 (got {args.train2_rate})")
  rates_sum = args.train1_rate + args.train2_rate
  if rates_sum >= 1.0:
    raise ValueError(f"The sum of train1-rate and train2-rate ({rates_sum}) must be less than 1.0")
  if args.skiprate < 1.0:
    raise ValueError(f"--skiprate must be 1.0 or greater (got {args.skiprate})")

  feature_set = features.get_feature_set_from_name(args.features)

  start_lambda = args.start_lambda or args.lambda_
  end_lambda = args.end_lambda or args.lambda_
  max_epoch = args.max_epochs or 800
  if args.resume_from_model is None:
    nnue = M.NNUE(feature_set=feature_set,
      start_lambda=start_lambda,
      max_epoch=max_epoch,
      end_lambda=end_lambda,
      gamma=args.gamma,
      lr=args.lr,
      epoch_size=args.epoch_size,
      batch_size=args.batch_size,
      in_scaling=args.in_scaling,
      out_scaling=args.out_scaling,
      offset=args.offset,
      offset1=args.offset1,
      offset2=args.offset2,
      adjust_loss=args.adjust_loss)
  else:

    # 「.pt」の場合
    if args.resume_from_model.endswith(".pt"):

      # 1. まず、現在の設定でモデルの「器（インスタンス）」を作る
      nnue = M.NNUE(feature_set=feature_set,
                    start_lambda=start_lambda,
                    max_epoch=max_epoch,
                    end_lambda=end_lambda,
                    gamma=args.gamma,
                    lr=args.lr,
                    epoch_size=args.epoch_size,
                    batch_size=args.batch_size,
                    in_scaling=args.in_scaling,
                    out_scaling=args.out_scaling,
                    offset=args.offset,
                    offset1=args.offset1,
                    offset2=args.offset2,
                    adjust_loss=args.adjust_loss)

      # 2. CPU 上で重みファイルをロード
      # A .pt resume source may contain a complete NNUE Python object rather
      # than only tensor weights.  PyTorch 2.6+ defaults torch.load() to
      # weights_only=True, so opt into object loading at this boundary only.
      checkpoint = torch.load(
          args.resume_from_model, map_location='cpu', weights_only=False)
      model_dict = nnue.state_dict()

      # checkpoint が NNUE オブジェクトそのものだった場合
      if hasattr(checkpoint, 'state_dict'):
          # オブジェクトから辞書形式を取り出す
          checkpoint_dict = checkpoint.state_dict()
      else:
          # すでに辞書形式（state_dict）だった場合
          checkpoint_dict = checkpoint

      # 3. checkpoint_dict を使って、形状が一致するものだけを抽出
      pretrained_dict = {
          k: v for k, v in checkpoint_dict.items() 
          if k in model_dict and v.shape == model_dict[k].shape
      }

      # 形状が合わないもの（Factorized化で入力数が増えた場合など）をログに出す
      for k in checkpoint_dict.keys():
          if k in model_dict and checkpoint_dict[k].shape != model_dict[k].shape:
              print(f"Skipping parameter {k} due to shape mismatch: {checkpoint_dict[k].shape} vs {model_dict[k].shape}")
          elif k not in model_dict:
              print(f"Parameter {k} not found in current model")

          if k in model_dict:
              if checkpoint_dict[k].shape == model_dict[k].shape:
                  model_dict[k].copy_(checkpoint_dict[k])

              elif k in ["input.weight", "input.v", "layer_stacks.phase_proj.weight", "layer_stacks.phase_proj.bias"]:
                  print(f"形状が異なりますが、重なっている部分だけコピーします: {k}")

                  old_shape = checkpoint_dict[k].shape
                  new_shape = model_dict[k].shape
                  print(f"Partial copy for {k}: {old_shape} -> {new_shape}")

                  # 1次元（Biasなど）か 2次元（Weightなど）かで処理を分ける
                  if len(new_shape) == 1:
                      # 共通する要素数分だけコピー
                      min_size = min(old_shape[0], new_shape[0])
                      model_dict[k][:min_size].copy_(checkpoint_dict[k][:min_size])
                  else:
                      # 2次元の場合（[出力, 入力]）
                      # 出力側(0次元目)と入力側(1次元目)の両方で共通する範囲を特定
                      min_dim0 = min(old_shape[0], new_shape[0])
                      min_dim1 = min(old_shape[1], new_shape[1])
                      model_dict[k][:min_dim0, :min_dim1].copy_(checkpoint_dict[k][:min_dim0, :min_dim1])

                  print(f"Completed partial copy for {k}")

      # 4. 現在のモデルの state_dict を更新してロード
      model_dict.update(pretrained_dict)
      nnue.load_state_dict(model_dict, strict=False)

    # 「.ckpt」の場合
    else:
      nnue = M.NNUE.load_from_checkpoint(args.resume_from_model, feature_set=feature_set, strict=False)

      """
      # 1. まず、新しい構造のモデルを普通に作る
      nnue = M.NNUE(feature_set=feature_set)
      
      # 2. チェックポイントを「ただの辞書」として読み込む
      checkpoint = torch.load(args.resume_from_model, map_location='cpu')
      state_dict = checkpoint["state_dict"]
      
      # 3. サイズが合わないパラメータを除外した新しい state_dict を作る
      new_state_dict = {}
      for k, v in state_dict.items():
          # モデル側の現在のパラメータ形状を取得
          if k in nnue.state_dict():
              target_shape = nnue.state_dict()[k].shape
              if v.shape == target_shape:
                  new_state_dict[k] = v
              else:
                  print(f"[Skip] {k}: shape mismatch (ckp: {v.shape} vs model: {target_shape})")
          else:
              print(f"[Skip] {k}: not in model")
      
      # 4. フィルタリングした重みを適用する
      nnue.load_state_dict(new_state_dict, strict=False)

      print("Load successful!")
      """

    nnue.set_feature_set(feature_set)
    nnue.in_scaling = args.in_scaling
    nnue.out_scaling = args.out_scaling
    nnue.offset = args.offset
    nnue.offset1 = args.offset1
    nnue.offset2 = args.offset2
    nnue.adjust_loss = args.adjust_loss
    nnue.start_lambda = start_lambda
    nnue.end_lambda = end_lambda
    nnue.max_epoch = max_epoch
    # we can set the following here just like that because when resuming
    # from .pt the optimizer is only created after the training is started
    nnue.gamma = args.gamma
    nnue.lr = args.lr

  print("Feature set: {}".format(feature_set.name))
  print("Num real features: {}".format(feature_set.num_real_features))
  print("Num virtual features: {}".format(feature_set.num_virtual_features))
  print("Num features: {}".format(feature_set.num_features))

  print("Training with {} and {} and {} validating with {}".format(args.train1, args.train2, args.train3, args.val))

  pl.seed_everything(args.seed)
  print("Seed {}".format(args.seed))

  if args.gpus < 0:
    raise ValueError(f"--gpus must be 0 or greater (got {args.gpus})")

  batch_size = args.batch_size
  if batch_size <= 0:
    batch_size = 128 if args.gpus == 0 else 8192
  print('Using batch size {}'.format(batch_size))

  print('Smart fen skipping: {}'.format(args.smart_fen_skipping))
  print('Random fen skipping: {}'.format(args.random_fen_skipping))

  if args.threads > 0:
    print('limiting torch to {} threads.'.format(args.threads))
    t_set_num_threads(args.threads)

  logdir = args.default_root_dir if args.default_root_dir else 'logs/'
  print('Using log dir {}'.format(logdir), flush=True)

  tb_logger = pl_loggers.TensorBoardLogger(logdir)
  if text_log_tee is not None:
    text_log_tee.bind_lightning_version(tb_logger.version)
    print(f"Text log: {text_log_tee.path}", flush=True)
  interrupt_controller = GracefulInterruptController()
  checkpoint_callback = NetworkSaveCheckpoint(
      every_n_epochs=args.network_save_period,
      log_dir=tb_logger.log_dir,
      interrupt_controller=interrupt_controller)
  trainer_device_args = (
      {"accelerator": "gpu", "devices": args.gpus}
      if args.gpus > 0
      else {"accelerator": "cpu", "devices": 1})
  trainer = pl.Trainer(
      callbacks=[checkpoint_callback],
      logger=tb_logger,
      max_epochs=args.max_epochs,
      default_root_dir=args.default_root_dir,
      log_every_n_steps=args.log_every_n_steps,
      **trainer_device_args)

  main_device = str(trainer.strategy.root_device)

  if args.py_data:
    print('Using python data loader')
    train, val = data_loader_py(args.train1, args.val, feature_set, batch_size, main_device)
  else:
    print('Using c++ data loader')
    train, val = data_loader_cc(args.train1, args.train2, args.train3, args.val, feature_set, args.num_workers, batch_size, args.smart_fen_skipping, args.random_fen_skipping, main_device, args.epoch_size, args.train1_rate, args.train2_rate, args.skiprate, args.mirror)

  torch.set_float32_matmul_precision('high')
  interrupt_controller.install()
  try:
    try:
      trainer.fit(nnue, train, val)
    except SystemExit:
      # Lightning 2.6 calls on_exception(KeyboardInterrupt), performs its own
      # graceful teardown, then raises SystemExit(1).  Once our callback has
      # successfully saved final.ckpt, translate only that known interrupt into
      # a normal process exit.  Other SystemExit causes must retain their code.
      if not checkpoint_callback.keyboard_interrupt_handled:
        raise
  finally:
    interrupt_controller.restore()

  print(f'tb_logger.log_dir={tb_logger.log_dir}')
  checkpoint_callback.save_final_checkpoint(trainer)
  if text_log_tee is not None:
    text_log_tee.close()

if __name__ == '__main__':
  main()
