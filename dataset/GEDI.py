import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, Optional, Callable
import numpy as np

class GEDIDataset(Dataset):
    def __init__(self, data_path: Union[str, Path, pd.DataFrame]):
        """
        Args:
            data_path: A path to the pickle file.
            waveform_transform: Optional transform to apply to the waveform arrays.
        """
        # Load the DataFrame if a path is provided.
        self.df = pd.DataFrame(pd.read_pickle(data_path))
        print(f"Loaded dataframe with {(self.df.shape[0])} rows")
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve the row from the DataFrame.
        row = self.df.iloc[idx]

        y = row['y_normalized']  # waveform y (normalized)
        y = torch.tensor(y, dtype=torch.float)
        #y = (y - 0.001953125) / 0.004507618 # already normalized in dataset
        y = y.unsqueeze(0) # (L, ) -> (1, L) # add channel, easier to work with conv1d

        # Get features for conditioning
        cond_col_mask = [5]
        condition_columns = np.array([
            'geolocation/latitude_bin0',            # 0
            'geolocation/longitude_bin0',           # 1
            'geolocation/elevation_bin0',           # 2 y
            'land_cover_data/landsat_treecover',    # 3
            'land_cover_data/modis_nonvegetated',   # 4
            'land_cover_data/modis_treecover'       # 5 y
        ])
        condition_ranges = np.array([
            [-90, 90],         # latitude 
            [-180, 180],       # longitude
            [0.0, 10000.0],    # elevation
            [0.0, 100.0],      # landsat treecover
            [0.0, 100.0],      # modis nonvegetated
            [0.0, 100.0]       # modis treecover
        ])
        
        condition_values = np.array([row[col] for col in condition_columns[cond_col_mask]], dtype=np.float32)
        conditions = torch.tensor(condition_values, dtype=torch.float)

        for i, cond_idx in enumerate(cond_col_mask):
            min, max = condition_ranges[cond_idx]
            range = max - min
            conditions[i] = ((conditions[i] - min) / range) # standardize to [0, 1]

        return y, conditions


def create_gedi_dataset(data_path: Union[str, Path, pd.DataFrame],
                        batch_size: int,
                        **kwargs):
    """
    Creates a DataLoader for the WaveformDataset.
    """
        
    dataset = GEDIDataset(data_path)

    subset_size = kwargs.get("subset", 1)
    if subset_size < 1:
        # use
        dataset_size = len(dataset)
        num_samples = int(subset_size * dataset_size)
        indices = list(range(dataset_size))
        subset_indices = np.random.choice(indices, size=num_samples, replace=False)
        sampler = torch.utils.data.SubsetRandomSampler(subset_indices)
        shuffle = None
    else:
        sampler = None
        shuffle = kwargs.get("shuffle", True)

    loader_params = dict(
        shuffle=shuffle,
        sampler=sampler,
        drop_last=kwargs.get("drop_last", False),
        pin_memory=kwargs.get("pin_memory", True),
        num_workers=kwargs.get("num_workers", 20),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, **loader_params)
    
    return dataloader
