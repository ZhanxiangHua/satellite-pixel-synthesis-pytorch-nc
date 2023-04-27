from io import BytesIO
import math
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import tensor_transforms as tt

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt


# idx =  0
# variables = ['Z','T','R']
# lon_min = 230 + 10 #230 - 5
# lon_max = 300 - 10 #300 + 5
# lat_min = 20 - 5 #20 - 5
# lat_max = 55 + 10 #55 + 10
# lon_bnds, lat_bnds = (lon_min, lon_max), (lat_max, lat_min)
# hr_ds = xr.open_dataset('../era_reanalysis/2020_01_hr.nc', engine = 'h5netcdf')[variables].isel(time = idx)
# t = pd.to_datetime(hr_ds['time'].values)
# days = (t - pd.to_datetime("2020-01-15 12:00:00")).dt.days
# means = hr_ds.mean(dim = ['latitude','longitude'])
# stds = hr_ds.std(dim = ['latitude','longitude'])
# hr_ds = (hr_ds - means) / stds

# lr_ds = hr_ds.sel(longitude=slice(*lon_bnds, 4), latitude=slice(*lat_bnds, 4))
# hr_ds = torch.tensor(hr_ds.to_array().to_numpy(), dtype=torch.float32)
# lr_ds = torch.tensor(lr_ds.to_array().to_numpy(), dtype=torch.float32) #.expand(1,-1,-1,-1)

class FMoWSentinel2(Dataset):
    # def standardize(input_tensor):
    #     means = input_tensor.mean(dim=(input_tensor.dim()-2,input_tensor.dim()-1), keepdim=True)
    #     stds = input_tensor.std((input_tensor.dim()-2,input_tensor.dim()-1), keepdim=True)
    #     return (input_tensor - means) / stds 
    
    def __init__(self, nc_path = '../era_reanalysis/2020_01_hr.nc', variables =  ['Z','T','R'], transform = None, enc_transform = None, resolution = 201, integer_values = None):
        """
        Args:
            nc_path (string): path to nc file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.variables = variables
        self.integer_values = integer_values
        self.resolution = resolution
        # Transforms
        self.transform = transform
        self.enc_transform = enc_transform
        # Lat Lon boundaries
        self.lon_min = 230 + 10 #230 - 5
        self.lon_max = 300 - 10 #300 + 5
        self.lat_min = 20 - 5 #20 - 5
        self.lat_max = 55 + 10 #55 + 10
        self.lon_bnds, self.lat_bnds = (self.lon_min, self.lon_max), (self.lat_max, self.lat_min)
        
        # open the netcdf file and standardize
        self.ds = xr.open_dataset(nc_path, engine = 'h5netcdf')[self.variables]
        self.means = self.ds.mean(dim = ['latitude','longitude'])
        self.stds = self.ds.std(dim = ['latitude','longitude'])
        self.ds = (self.ds - self.means) / self.stds
        
        # Calculate len
        self.time = self.ds.time
        self.data_len = len(self.time)

    def __getitem__(self, index):
        date_anchor = pd.to_datetime("2020-01-15 12:00:00")
        t_stamp = pd.to_datetime(self.ds['time'].isel(time = index).values)
        t = (t_stamp - date_anchor)/pd.to_timedelta(1, unit='D')
        
        high = self.ds.isel(time = index)
        low = high.sel(longitude=slice(*self.lon_bnds, 4), latitude=slice(*self.lat_bnds, 4)).copy()
        
        if index  == 0:
            hr2_t_idx = index
        elif index < 12:
            hr2_t_idx = torch.randint(0, index, (1,))[0]
        else:
            hr2_t_idx = torch.randint(index - 12, index, (1,))[0]
        
        high2 = self.ds.isel(time = hr2_t_idx).copy()
        
        # convert to tensor
        high = torch.tensor(high.to_array().to_numpy(), dtype=torch.float32).clamp(min=-2, max=2)
        high2 = torch.tensor(high2.to_array().to_numpy(), dtype=torch.float32).clamp(min=-2, max=2)
        low = torch.tensor(low.to_array().to_numpy(), dtype=torch.float32).clamp(min=-2, max=2)
        # Transform the image
        # high = self.transform(high)
        # high2 = self.transform(high2)
        low = self.enc_transform(low)
        coords = tt.convert_to_coord_uneven_t(1, self.resolution, self.resolution, t, integer_values=self.integer_values)
        #print(coords)
        high = torch.cat([high, coords], 0)

        return (high, low, high2)
    
    def __len__(self):
        return self.data_len    

if __name__ == "__main__":
    g = torch.Generator()
    g.manual_seed(0)
    
    enc_transform = transforms.Compose(
        [
            transforms.Resize((201,201), interpolation=transforms.InterpolationMode.BILINEAR, antialias = False),
            #transforms.CenterCrop(256),
            #transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    transform = transforms.Compose(
        [
            transforms.Resize((256,256)),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    dataset = FMoWSentinel2(nc_path = '../era_reanalysis/2020_01_hr.nc', transform = None, enc_transform=enc_transform, resolution=201, integer_values=False)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # for step, (high, low, high2) in enumerate(train_dataloader):
    #     print(high.shape)
    #     print(low.shape)
    #     print(high2.shape)
    #     break
    

    # Display image and label.
    high, low, high2 = next(iter(train_dataloader))
    print(f"target_hr: {high.size()}")
    print(f"target_lr: {low.size()}")
    print(f"hr_t': {high2.size()}")
    
    target_hr_sample = high[0,0:3,:,:].squeeze()
    target_lr_sample = low[0].squeeze()
    hr_t_prime_sample = high2[0].squeeze()
    
    plt.imshow(hr_t_prime_sample[1,:,:], cmap=plt.cm.Blues)
    plt.imshow(target_hr_sample[1,:,:], cmap=plt.cm.Blues)
    plt.imshow(target_lr_sample[1,:,:], cmap=plt.cm.Blues)
    
    plt.show()
    