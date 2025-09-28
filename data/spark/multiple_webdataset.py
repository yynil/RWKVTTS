import os
import json
import torch
import torchaudio
import numpy as np
from datasets import load_dataset, Audio
from typing import Dict, List
from pathlib import Path
import tarfile
import logging
import random
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultipleWebDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str = None,
        data_files: List[str] = None,
        target_sr: int = 16000,
        target_channels: int = 1,
        shuffle: bool = True,
        verify_tar: bool = False
    ):
        self.target_sr = target_sr
        self.target_channels = target_channels
        self.verify_tar = verify_tar
        self.datasets = []

        if data_files:
            files_to_load = data_files
        elif data_dir:
            data_path = Path(data_dir)
            if data_path.is_file():
                files_to_load = [str(data_path)]
            else:
                files_to_load = [str(f) for f in data_path.glob("**/*.tar*")]
        else:
            raise ValueError("Either data_dir or data_files must be provided.")

        for file_path in files_to_load:
            if not self._is_valid_tar(Path(file_path)):
                logger.warning(f"Skipping corrupted or invalid tar file: {file_path}")
                continue
            try:
                dataset = load_dataset("webdataset", data_files=[file_path], split="train")
                features = dataset.features
                audio_key = next((key for key, value in features.items() if isinstance(value, Audio)), None)

                if audio_key is None:
                    logger.warning(f"No audio data found in {file_path}. Skipping.")
                    continue

                dataset = dataset.cast_column(audio_key, Audio(sampling_rate=target_sr, mono=(target_channels == 1)))
                if audio_key != "audio":
                    dataset = dataset.rename_column(audio_key, "audio")
                
                self.datasets.append(dataset)
                logger.info(f"Successfully loaded dataset: {file_path} with {len(dataset)} samples.")
            except Exception as e:
                logger.error(f"Error loading dataset {file_path}: {e}")

        if not self.datasets:
            raise ValueError("No valid datasets were loaded.")

        self.cumulative_sizes = np.cumsum([len(d) for d in self.datasets])
        self.total_size = self.cumulative_sizes[-1] if len(self.cumulative_sizes) > 0 else 0
        
        self.shuffle = shuffle
        self.indices = np.arange(self.total_size)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _is_valid_tar(self, tar_path: Path) -> bool:
        if not self.verify_tar:
            return True
        try:
            with tarfile.open(tar_path, 'r:*') as tar:
                tar.getmembers()
            return True
        except tarfile.ReadError:
            logger.warning(f"Tar file is corrupted: {tar_path}")
            return False
        except Exception as e:
            logger.warning(f"Failed to validate tar file {tar_path}: {e}")
            return False

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, idx: int) -> Dict:
        if idx >= self.total_size:
            raise IndexError(f"Index {idx} is out of bounds for dataset with size {self.total_size}")

        original_idx = self.indices[idx] if self.shuffle else idx
        dataset_idx = np.searchsorted(self.cumulative_sizes, original_idx, side='right')
        sample_idx = original_idx - (self.cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else 0)

        retry_times = 0
        while retry_times < 3:
            try:
                return self.datasets[dataset_idx][int(sample_idx)]
            except Exception as e:
                print(f"Error in __getitem__: {e}")
                print(f"Attempting to access idx: {idx} (original: {original_idx}), which maps to dataset {dataset_idx}, sample {sample_idx}")
                retry_times += 1
                time.sleep(1)
        
        raise Exception(f"Failed to retrieve item after 3 retries for index: {idx}")
