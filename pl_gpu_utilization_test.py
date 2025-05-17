#!/usr/bin/env python
# lightning_gpu_test_console_stats.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor # Removed DeviceStatsMonitor
from pytorch_lightning.tuner import Tuner
from pytorch_lightning.loggers import TensorBoardLogger # Still useful for other metrics

# Try to import pynvml for the custom callback
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("WARNING: pynvml library not found. GPU stats will not be printed by ConsoleGpuStatsCallback.")
    print("Please install with: pip install pynvml")

# ------------------------- Custom GPU Stats Callback for Console -------------------------
class ConsoleGpuStatsCallback(pl.Callback):
    def __init__(self, device_id: int = 0, print_interval_batches: int = 50):
        super().__init__()
        self.device_id = device_id
        self.print_interval_batches = print_interval_batches
        self.handle = None
        self.initialized_successfully = False

        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                self.initialized_successfully = True
                print(f"ConsoleGpuStatsCallback initialized for GPU {self.device_id}.")
            except Exception as e:
                print(f"ERROR: ConsoleGpuStatsCallback: Error initializing pynvml or getting GPU handle: {e}")
                self.handle = None
        else:
            print("ConsoleGpuStatsCallback: pynvml not available, cannot monitor GPU stats.")

    def _get_stats_str(self) -> str:
        if not self.handle or not self.initialized_successfully:
            return "N/A (pynvml not initialized or GPU handle error)"
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            # Power reporting might not be supported on all GPUs/systems or require specific permissions
            try:
                power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # In Watts
                power_str = f"{power:.1f}W"
            except pynvml.NVMLError as e:
                power_str = "N/A"


            return (f"GPU[{self.device_id}] Util: {util.gpu}%, MemUtil: {util.memory}%, "
                    f"MemUsed: {mem_info.used / (1024**2):.1f}MB / {mem_info.total / (1024**2):.1f}MB, "
                    f"Temp: {temp}°C, Power: {power_str}")
        except Exception as e:
            return f"Error querying GPU stats: {e}"

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int):
        if not self.initialized_successfully:
            return
        # Only print from the main process (global_rank 0)
        if trainer.is_global_zero and (batch_idx + 1) % self.print_interval_batches == 0:
            stats_str = self._get_stats_str()
            # Using trainer.print to integrate with Lightning's console output
            trainer.print(f"  GPU Stats (batch {batch_idx + 1}/{trainer.num_training_batches}): {stats_str}")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if not self.initialized_successfully:
            return
        if trainer.is_global_zero:
            stats_str = self._get_stats_str()
            trainer.print(f"  GPU Stats (End of Train Epoch {trainer.current_epoch + 1}): {stats_str}")

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self.initialized_successfully and PYNVML_AVAILABLE: # Check PYNVML_AVAILABLE again
            try:
                pynvml.nvmlShutdown()
                print("ConsoleGpuStatsCallback: pynvml shutdown.")
            except Exception as e:
                print(f"ERROR: ConsoleGpuStatsCallback: Error shutting down pynvml: {e}")
    
    def on_exception(self, trainer: pl.Trainer, pl_module: pl.LightningModule, exception: BaseException):
        # Ensure nvmlShutdown is called if an exception occurs during training
        if self.initialized_successfully and PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
                print("ConsoleGpuStatsCallback: pynvml shutdown due to exception.")
            except Exception as e:
                print(f"ERROR: ConsoleGpuStatsCallback: Error shutting down pynvml during exception: {e}")


# ------------------------- Model Definitions -------------------------
class SimpleConvNet(nn.Module):
    """A simple convolutional neural network for image classification"""
    def __init__(self, num_classes=10, channels=3):
        super(SimpleConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class LargeConvNet(nn.Module):
    """A larger convolutional neural network for image classification"""
    def __init__(self, num_classes=10, channels=3):
        super(LargeConvNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), 
            nn.Flatten(),
            nn.Linear(512, 1024), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ------------------------- LightningDataModule -------------------------
class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str = 'cifar10', data_dir: str = './data',
                 batch_size: int = 128, num_workers: int = 4,
                 pin_memory: bool = True, prefetch_factor: int = 2,
                 persistent_workers: bool = True):
        super().__init__()
        self.save_hyperparameters()
        self._set_transforms_and_dims()

    def _set_transforms_and_dims(self):
        if self.hparams.dataset_name == 'cifar10':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.num_classes = 10
            self.channels = 3
        # ... (rest of dataset configs for cifar100, mnist as before) ...
        elif self.hparams.dataset_name == 'cifar100':
            self.transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            self.transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
            self.num_classes = 100
            self.channels = 3
        elif self.hparams.dataset_name == 'mnist':
            self.transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
            self.transform_test = self.transform_train
            self.num_classes = 10
            self.channels = 1
        else:
            raise ValueError(f"Unsupported dataset: {self.hparams.dataset_name}")


    def prepare_data(self):
        if self.hparams.dataset_name == 'cifar10':
            torchvision.datasets.CIFAR10(self.hparams.data_dir, train=True, download=True)
            torchvision.datasets.CIFAR10(self.hparams.data_dir, train=False, download=True)
        # ... (rest of prepare_data for cifar100, mnist) ...
        elif self.hparams.dataset_name == 'cifar100':
            torchvision.datasets.CIFAR100(self.hparams.data_dir, train=True, download=True)
            torchvision.datasets.CIFAR100(self.hparams.data_dir, train=False, download=True)
        elif self.hparams.dataset_name == 'mnist':
            torchvision.datasets.MNIST(self.hparams.data_dir, train=True, download=True)
            torchvision.datasets.MNIST(self.hparams.data_dir, train=False, download=True)


    def setup(self, stage: str = None):
        if stage == 'fit' or stage is None:
            if self.hparams.dataset_name == 'cifar10':
                full_dataset = torchvision.datasets.CIFAR10(self.hparams.data_dir, train=True, transform=self.transform_train)
            # ... (rest of setup 'fit' for cifar100, mnist) ...
            elif self.hparams.dataset_name == 'cifar100':
                full_dataset = torchvision.datasets.CIFAR100(self.hparams.data_dir, train=True, transform=self.transform_train)
            elif self.hparams.dataset_name == 'mnist':
                full_dataset = torchvision.datasets.MNIST(self.hparams.data_dir, train=True, transform=self.transform_train)
            else: 
                raise ValueError(f"Dataset {self.hparams.dataset_name} not configured in setup.")
            train_size = int(0.9 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

        if stage == 'test' or stage is None:
            if self.hparams.dataset_name == 'cifar10':
                self.test_dataset = torchvision.datasets.CIFAR10(self.hparams.data_dir, train=False, transform=self.transform_test)
            # ... (rest of setup 'test' for cifar100, mnist) ...
            elif self.hparams.dataset_name == 'cifar100':
                self.test_dataset = torchvision.datasets.CIFAR100(self.hparams.data_dir, train=False, transform=self.transform_test)
            elif self.hparams.dataset_name == 'mnist':
                self.test_dataset = torchvision.datasets.MNIST(self.hparams.data_dir, train=False, transform=self.transform_test)
            else: 
                raise ValueError(f"Dataset {self.hparams.dataset_name} not configured in setup for test.")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
                          shuffle=True, 
                          persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False,
                          prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False,
                          prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
                          persistent_workers=self.hparams.persistent_workers if self.hparams.num_workers > 0 else False,
                          prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None)

# ------------------------- LightningModule -------------------------
class ImageClassifierSystem(pl.LightningModule):
    def __init__(self, model_name: str = 'simple', num_classes: int = 10, channels: int = 3,
                 learning_rate: float = 1e-3, compile_model: bool = False):
        super().__init__()
        self.save_hyperparameters()

        if model_name == 'simple':
            self.model_arch = SimpleConvNet(num_classes=num_classes, channels=channels)
        elif model_name == 'large':
            self.model_arch = LargeConvNet(num_classes=num_classes, channels=channels)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        if compile_model and hasattr(torch, 'compile'):
            print("Compiling the model with torch.compile()...")
            try:
                self.model_arch = torch.compile(self.model_arch)
            except Exception as e:
                print(f"torch.compile failed: {e}. Using uncompiled model.")
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model_arch(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', acc, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1) # Removed verbose
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

# ------------------------- Main Function -------------------------
def main(args):
    # Set matmul precision for Tensor Cores on Ampere+ GPUs
    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        print("Setting float32 matmul precision to 'high' for Tensor Cores.")
        torch.set_float32_matmul_precision('high') # or 'medium'
        
    pl.seed_everything(42, workers=True)

    datamodule = ImageNetDataModule(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        data_dir=args.data_dir,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers
    )
    datamodule.prepare_data()
    datamodule.setup('fit') 

    model_system = ImageClassifierSystem(
        model_name=args.model,
        num_classes=datamodule.num_classes,
        channels=datamodule.channels,
        learning_rate=args.lr,
        compile_model=args.compile_model
    )

    logger = TensorBoardLogger("tb_logs", name=f"{args.dataset}_{args.model}")
    
    callbacks = [LearningRateMonitor(logging_interval='epoch')]
    # Add our custom console GPU stats printer instead of DeviceStatsMonitor
    if torch.cuda.is_available() and args.gpus > 0:
        console_gpu_stats_callback = ConsoleGpuStatsCallback(print_interval_batches=args.print_gpu_stats_interval)
        callbacks.append(console_gpu_stats_callback)
    # Note: Removed DeviceStatsMonitor as per request to print instead of log to TB for these stats.
    # If you still want DeviceStatsMonitor for TB, you can add it back, but pynvml init/shutdown might conflict
    # if not handled carefully (e.g., one global pynvml manager).
    # For simplicity, using only one method of pynvml access here.

    trainer = pl.Trainer(
        accelerator="gpu" if args.gpus > 0 and torch.cuda.is_available() else "cpu",
        devices=args.gpus if args.gpus > 0 and torch.cuda.is_available() else "auto",
        max_epochs=args.epochs,
        precision="16-mixed" if args.amp else "32-true",
        callbacks=callbacks,
        logger=logger,
        deterministic=False, 
        benchmark=True,    
    )

    if args.auto_scale_batch_size and args.gpus > 0 and torch.cuda.is_available():
        print(f"Attempting to auto-scale batch size from initial: {datamodule.hparams.batch_size}...")
        tuner = Tuner(trainer)
        try:
            tuner.scale_batch_size(model_system, datamodule=datamodule, mode="power")
            print(f"Auto-scaled batch size to: {datamodule.hparams.batch_size}")
            if hasattr(datamodule, 'batch_size'): # Ensure direct attribute is also updated
                 datamodule.batch_size = datamodule.hparams.batch_size
        except Exception as e:
            print(f"Failed to auto-scale batch size: {e}. Using initial batch size: {args.batch_size}")
            datamodule.hparams.batch_size = args.batch_size
            if hasattr(datamodule, 'batch_size'):
                datamodule.batch_size = args.batch_size

    print(f"\nStarting training with effective configuration:")
    print(f"  Dataset: {datamodule.hparams.dataset_name}")
    print(f"  Model: {model_system.hparams.model_name}")
    print(f"  Batch Size: {datamodule.hparams.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Workers: {datamodule.hparams.num_workers}")
    print(f"  AMP (16-mixed): {args.amp}")
    print(f"  Compile Model: {args.compile_model and hasattr(torch, 'compile')}")
    print(f"  Device: {trainer.accelerator} with {trainer.num_devices} device(s)\n")

    try:
        trainer.fit(model_system, datamodule=datamodule)
        print("\nTraining complete. Testing...")
        datamodule.setup('test') 
        trainer.test(model_system, datamodule=datamodule)
    except Exception as e:
        print(f"An error occurred during training or testing: {e}")
    finally:
        # Explicitly call the on_fit_end of our callback if trainer.fit did not complete
        # or if we want to ensure pynvml is shutdown.
        # This might be redundant if trainer.fit completes and calls it.
        # However, to be safe with external resources like pynvml:
        if 'console_gpu_stats_callback' in locals() and console_gpu_stats_callback.initialized_successfully:
            console_gpu_stats_callback.on_fit_end(trainer, model_system)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Lightning GPU Utilization Test with Console Stats')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist'])
    parser.add_argument('--model', type=str, default='simple', choices=['simple', 'large'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--persistent-workers', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--amp', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--auto-scale-batch-size', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--compile-model', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--print-gpu-stats-interval', type=int, default=50,
                        help='How often (in batches) to print GPU stats to console (default: 50)')


    args = parser.parse_args()
    
    if args.gpus == -1 and torch.cuda.is_available():
        args.gpus = torch.cuda.device_count()
    elif not torch.cuda.is_available() and args.gpus > 0:
        print("Warning: CUDA not available, requested GPUs will not be used. Switching to CPU.")
        args.gpus = 0
    
    main(args)