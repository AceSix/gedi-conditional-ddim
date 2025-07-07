# -*- coding:utf-8 -*-
###################################################################
###   @FilePath: \gedi-conditional-ddim\dataset\GEDI.py
###   @Author: AceSix
###   @Date: 2025-07-06 18:15:39
###   @LastEditors: AceSix
###   @LastEditTime: 2025-07-07 02:22:31
###   @Copyright (C) 2025 Brown U. All rights reserved.
###################################################################
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Union, Optional, Callable
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

train_keys = [
            'geolocation/latitude_bin0', 'geolocation/longitude_bin0',
            'AnnualTemp', 'MeanDiurnal', 'Isothermality', 
            'TempSeasonality', 'WarmestMTemp', 'ColdestMTemp', 'TempRange', 'WettestQTemp', 'DriestQTemp', 'WarmestQTemp', 'ColdestQTemp', 
            'AnnualPrec', 'WettestMPrec', 'DriestMPrec', 'PrecSeasonality', 'WettestQPrec', 'DriestQPrec', 'WarmestQPrec', 'ColdestQPrec', 
            'month_srad', 'month_vapr', 'month_wind',
            'roughness', 'slope', 'elevation', 'lulc_human', 
            'bdod_0-5cm_mean', 'bdod_100-200cm_mean', 'bdod_15-30cm_mean', 'bdod_30-60cm_mean', 'bdod_5-15cm_mean', 'bdod_60-100cm_mean', 
            'cec_0-5cm_mean', 'cec_100-200cm_mean', 'cec_15-30cm_mean', 'cec_30-60cm_mean', 'cec_5-15cm_mean', 'cec_60-100cm_mean', 
            'cfvo_0-5cm_mean', 'cfvo_100-200cm_mean', 'cfvo_15-30cm_mean', 'cfvo_30-60cm_mean', 'cfvo_5-15cm_mean', 'cfvo_60-100cm_mean', 
            'clay_0-5cm_mean', 'clay_100-200cm_mean', 'clay_15-30cm_mean', 'clay_30-60cm_mean', 'clay_5-15cm_mean', 'clay_60-100cm_mean', 
            'nitrogen_0-5cm_mean', 'nitrogen_100-200cm_mean', 'nitrogen_15-30cm_mean', 'nitrogen_30-60cm_mean', 'nitrogen_5-15cm_mean', 'nitrogen_60-100cm_mean', 
            'ocd_0-5cm_mean', 'ocd_100-200cm_mean', 'ocd_15-30cm_mean', 'ocd_30-60cm_mean', 'ocd_5-15cm_mean', 'ocd_60-100cm_mean', 
            'phh2o_0-5cm_mean', 'phh2o_100-200cm_mean', 'phh2o_15-30cm_mean', 'phh2o_30-60cm_mean', 'phh2o_5-15cm_mean', 'phh2o_60-100cm_mean', 
            'sand_0-5cm_mean', 'sand_100-200cm_mean', 'sand_15-30cm_mean', 'sand_30-60cm_mean', 'sand_5-15cm_mean', 'sand_60-100cm_mean', 
            'silt_0-5cm_mean', 'silt_100-200cm_mean', 'silt_15-30cm_mean', 'silt_30-60cm_mean', 'silt_5-15cm_mean', 'silt_60-100cm_mean', 
            'soc_0-5cm_mean', 'soc_100-200cm_mean', 'soc_15-30cm_mean', 'soc_30-60cm_mean', 'soc_5-15cm_mean', 'soc_60-100cm_mean', 
        ]


class GEDIDataset(Dataset):
    def __init__(self, data_path: Union[str, Path, pd.DataFrame]):
        """
        Args:
            data_path: A path to the pickle file.
            waveform_transform: Optional transform to apply to the waveform arrays.
        """
        # Load the DataFrame if a path is provided.
        with open("../data/Segments/global_rh_prop_10kpp_p0.pkl", 'rb') as handle:
            _, props = pickle.load(handle)

        features = np.array([[p[k] for k in train_keys] for p in props])

        self.data_scaler = StandardScaler()
        self.data_scaler.fit(features)

        with open(data_path, 'rb') as handle:
            self.rhs, self.props = pickle.load(handle)
        
    def resample_array(self, arr, new_size):
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, new_size)
        return np.interp(x_new, x_old, arr)
    
    def __len__(self):
        return len(self.rhs)

    def __getitem__(self, idx):
        # Retrieve the row from the DataFrame.
        rh = self.rhs[idx]
        prop = self.props[idx]

        rh_resampled = self.resample_array(rh, 96)

        y = torch.tensor(rh_resampled, dtype=torch.float)
        y = y.unsqueeze(0) # (L, ) -> (1, L) # add channel, easier to work with conv1d

        # Get features for conditioning
        
        
        condition_values = np.array([prop[col] for col in train_keys], dtype=np.float32)
        conditions = torch.tensor(self.data_scaler.transform(condition_values[np.newaxis, :]), dtype=torch.float)[0]
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
