from io import BytesIO
import math
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional as tvF
import tensor_transforms as tt

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

# create custom class transform
class RC(transforms.RandomCrop):
    def __call__(self, img1, img2, img3):
        assert img1.size() == img2.size() == img3.size()
        # fix parameter
        i, j, h, w = self.get_params(img1, self.size)
        # return the image with the same transformation
        return [tvF.crop(img1, i, j, h, w), 
                tvF.crop(img2, i, j, h, w),
                tvF.crop(img3, i, j, h, w)]
# imgInput, imgTarget = RC(output_size=128)(imgInput, imgTarget)
# idx =  0
# variables = ['U','V']
# lon_min = 230 + 10 #230 - 5
# lon_max = 300 - 10 #300 + 5
# lat_min = 20 - 5 #20 - 5
# lat_max = 55 + 10 #55 + 10
# lon_bnds, lat_bnds = (lon_min, lon_max), (lat_max, lat_min)
# hr_ds = xr.open_dataset('../era_reanalysis/2020_01_hr.nc', engine = 'h5netcdf')[variables]#.isel(time = idx)
# lr_ds = hr_ds.coarsen(latitude=4, longitude=4, boundary='trim').mean()
# # t = pd.to_datetime(hr_ds['time'].values)
# # #days = (t - pd.to_datetime("2020-01-15 12:00:00")).dt.days

# hourly_grouped_mean = hr_ds.groupby('time.hour').mean(dim = ['time','latitude','longitude'])
# hourly_grouped_stds = hr_ds.groupby('time.hour').std(dim = ['time','latitude','longitude'])
# hr_ds_standardized_hourly = (hr_ds.groupby('time.hour') - hourly_grouped_mean).groupby('time.hour') / hourly_grouped_stds
# lr_ds_standardized_hourly = (lr_ds.groupby('time.hour') - hourly_grouped_mean).groupby('time.hour') / hourly_grouped_stds
# #check_bound = lr_ds_standardized_hourly.U.where(lr_ds_standardized_hourly.U >= 3).notnull().to_array().to_numpy()
# #count_overbound = np.count_nonzero(check_bound == True)
#lr_ds = hr_ds.sel(longitude=slice(*lon_bnds, 4), latitude=slice(*lat_bnds, 4)) # skip sampling


class FMoWSentinel2(Dataset):
    def __init__(self, nc_path = '/mnt/h/era_reanalysis/2020_01_hr.nc', variables = None, transform = None, enc_transform = None, resolution = None, integer_values = None):
        """
        Args:
            nc_path (string): path to nc file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.nc_path = nc_path
        self.variables = variables
        self.data_channels = len(variables)
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
        self.hr_ds = xr.open_dataset(self.nc_path, engine = 'h5netcdf')[self.variables]
        self.lr_ds = self.hr_ds.coarsen(latitude=10, longitude=10, boundary='trim').mean()
        
        self.hourly_grouped_mean =  self.hr_ds.groupby('time.hour').mean(dim = ['time','latitude','longitude'])
        self.hourly_grouped_stds =  self.hr_ds.groupby('time.hour').std(dim = ['time','latitude','longitude'])
        self.hr_ds_standardized_hourly = (self.hr_ds.groupby('time.hour') - self.hourly_grouped_mean).groupby('time.hour') / self.hourly_grouped_stds
        self.lr_ds_standardized_hourly = (self.lr_ds.groupby('time.hour') - self.hourly_grouped_mean).groupby('time.hour') / self.hourly_grouped_stds
        
        # Calculate len
        self.time = self.hr_ds.time
        self.data_len = len(self.time)

    def __getitem__(self, index):
        #date_anchor = pd.to_datetime("2020-01-15 12:00:00")
        t_stamp = pd.to_datetime(self.hr_ds_standardized_hourly['time'].isel(time = index).values)
        #t = (t_stamp - date_anchor)/pd.to_timedelta(1, unit='D')
        
        high =  self.hr_ds_standardized_hourly.isel(time = index)
        low = self.lr_ds_standardized_hourly.isel(time = index)
        #low = high.sel(longitude=slice(*self.lon_bnds, 4), latitude=slice(*self.lat_bnds, 4)).copy()
        
        if index  == 0:
            hr2_t_idx = index
        elif index < 12:
            hr2_t_idx = torch.randint(0, index, (1,))[0]
        else:
            hr2_t_idx = torch.randint(index - 12, index, (1,))[0]
        
        high2 = self.hr_ds_standardized_hourly.isel(time = hr2_t_idx).copy()
        
        high2_t_stamp = pd.to_datetime(high2['time'].values)
        #how is this t define in testing?
        t = (t_stamp - high2_t_stamp)/pd.to_timedelta(1, unit='H') #use time delta hours to lr
        # convert to tensor
        high = torch.tensor(high.to_array().to_numpy(), dtype=torch.float32).clamp(min=-4, max=4)
        high2 = torch.tensor(high2.to_array().to_numpy(), dtype=torch.float32).clamp(min=-4, max=4)
        low = torch.tensor(low.to_array().to_numpy(), dtype=torch.float32).clamp(min=-4, max=4)
        # Transform the image
        high = self.transform(high)
        high2 = self.transform(high2)
        low = self.enc_transform(low)
        
        high, high2, low = RC(size=self.resolution)(high, high2, low)
        
        #low = torch.where(low >= 0, low, 0) #filter out any negative value in Z,T,R
        coords = tt.convert_to_coord_uneven_t(1, self.resolution, self.resolution, t, integer_values=self.integer_values)
        #print(coords)
        high = torch.cat([high, coords], 0)

        #add additional return of high(time of day, channel, mean, stdv) and high2(time of day, channel, mean, stdv)       
        high_mean = torch.tensor(self.hourly_grouped_mean.sel(hour=t_stamp.hour).to_array().to_numpy(), dtype=torch.float32)
        high_stdv = torch.tensor(self.hourly_grouped_stds.sel(hour=t_stamp.hour).to_array().to_numpy(), dtype=torch.float32)
        high_norm_param = torch.stack((high_mean, high_stdv), dim = 1)
        
        high2_mean = torch.tensor(self.hourly_grouped_mean.sel(hour=high2_t_stamp.hour).to_array().to_numpy(), dtype=torch.float32)
        high2_stdv = torch.tensor(self.hourly_grouped_stds.sel(hour=high2_t_stamp.hour).to_array().to_numpy(), dtype=torch.float32)
        high2_norm_param = torch.stack((high2_mean, high2_stdv), dim = 1) #hourly([[u_mean, v_mean],[u_stds, v_stds]])

        return (high, low, high2, high_norm_param, high2_norm_param)
    
    def __len__(self):
        return self.data_len    

if __name__ == "__main__":
    g = torch.Generator()
    g.manual_seed(0)
    
    enc_transform = transforms.Compose(
        [
            transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BILINEAR, antialias = False),
            #transforms.RandomCrop(128),
            #transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    transform = transforms.Compose(
        [
            transforms.Resize((256,256), interpolation=transforms.InterpolationMode.BILINEAR, antialias = False),
            #transforms.RandomCrop(128),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    
    selected_RC_resolution = 128
    selected_vars = ['U','V']
    
    dataset = FMoWSentinel2(nc_path = '/mnt/h/era_reanalysis/2020_01_hr.nc', variables = selected_vars, transform = transform,enc_transform=enc_transform, 
                            resolution = selected_RC_resolution, integer_values=False)
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, generator=g) 

    # Display image and label.
    high, low, high2, high_norm_param, high2_norm_param = next(iter(train_dataloader))
    print(f"target_hr: {high.size()}")
    print(f"target_lr: {low.size()}")
    print(f"hr_t': {high2.size()}")
    
    target_hr_sample = high[0,0:len(selected_vars),:,:].squeeze() #second vector change with number of data channels
    target_lr_sample = low[0].squeeze()
    hr_t_prime_sample = high2[0].squeeze()
    
    #test inverse scaling back to the original magnitude
    unnorm_low = low * high_norm_param[:,:,1].view(-1, len(selected_vars), 1, 1) + high_norm_param[:,:,0].view(-1, len(selected_vars), 1, 1)
    unnorm_high = high[:,0:len(selected_vars),:,:] * high_norm_param[:,:,1].view(-1, len(selected_vars), 1, 1) + high_norm_param[:,:,0].view(-1, len(selected_vars), 1, 1)
    unnorm_high2 = high2 * high2_norm_param[:,:,1].view(-1, len(selected_vars), 1, 1) + high2_norm_param[:,:,0].view(-1, len(selected_vars), 1, 1)
    
    #normalized
    plt.imshow(hr_t_prime_sample[1,:,:], cmap=plt.cm.coolwarm, vmin = -4, vmax = 4)
    plt.imshow(target_hr_sample[1,:,:], cmap=plt.cm.coolwarm, vmin = -4, vmax = 4)
    plt.imshow(target_lr_sample[1,:,:], cmap=plt.cm.coolwarm, vmin = -4, vmax = 4)
    
    #unnormalized
    plt.imshow(unnorm_high2[0,1,:,:], cmap=plt.cm.coolwarm, vmin = -40, vmax = 40)
    plt.imshow(unnorm_high[0,1,:,:], cmap=plt.cm.coolwarm, vmin = -40, vmax = 40)
    plt.imshow(unnorm_low[0,1,:,:], cmap=plt.cm.coolwarm, vmin = -40, vmax = 40)
    
    
    
    
    
    #save a gridded test image
    grid_img = utils.make_grid(low[0:8,:,:,:], nrow=int(8 ** 0.5))
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=-4, vmax=4)
    image = cmap(norm(grid_img[0,:,:]))
    plt.imsave('test.png', image)
    
    # transformation = transforms.ToPILImage()
    # img = transformation(target_lr_sample[1:2,:,:])
    # img.show()
    
    plt.show()
    