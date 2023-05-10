from io import BytesIO
import math
import lmdb
import glob
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.transforms import functional as tvF
import tensor_transforms as tt
from torchmetrics import PeakSignalNoiseRatio

import numpy as np
import pandas as pd
import xarray as xr
import dask
from matplotlib import pyplot as plt

# from dask.distributed import Client, LocalCluster

# cluster = LocalCluster(n_workers=6, memory_limit = '2GiB')
# client = Client(cluster)
# print(cluster.dashboard_link)
def save_single_channel_visual(batch_data, nrow, channel, l1_loss, psnr_score):
    grid_img = utils.make_grid(batch_data, nrow=nrow)
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=-4, vmax=4)
    text = plt.text(grid_img.shape[1]//2, grid_img.shape[0]//2, s = f'psnr = {round(psnr_score,3)} \n l1 = {round(l1_loss,5)}')
    image = cmap(norm(grid_img[channel,:,:].cpu()))
    return image

# create custom class transform
class RC(transforms.RandomCrop):
    def __call__(self, img1, img2, img3):
        assert img1.size() == img2.size() == img3.size()
        # fix parameter
        i, j, h, w = self.get_params(img1, self.size)
        #print(i,j,h,w)
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
# file_paths = glob.glob('/mnt/h/era_reanalysis/*_hr.nc')
# hr_ds = xr.open_mfdataset(file_paths, engine = 'h5netcdf', parallel = True, chunks='auto')[variables]#.isel(time = idx)
# lr_ds = hr_ds.coarsen(latitude=4, longitude=4, boundary='trim').mean()

# hourly_grouped_mean = xr.open_dataset('/mnt/h/era_reanalysis/2016-2022_01_hourly_mean.nc', engine='h5netcdf')
# hourly_grouped_stds = xr.open_dataset('/mnt/h/era_reanalysis/2016-2022_01_hourly_stds.nc', engine='h5netcdf')

# hr_ds_standardized_hourly = ((hr_ds.groupby('time.hour') - hourly_grouped_mean).groupby('time.hour') / hourly_grouped_stds).compute()
# lr_ds_standardized_hourly = ((lr_ds.groupby('time.hour') - hourly_grouped_mean).groupby('time.hour') / hourly_grouped_stds).compute()

# # t = pd.to_datetime(hr_ds['time'].values)
# # #days = (t - pd.to_datetime("2020-01-15 12:00:00")).dt.days

# #check_bound = lr_ds_standardized_hourly.U.where(lr_ds_standardized_hourly.U >= 3).notnull().to_array().to_numpy()
# #count_overbound = np.count_nonzero(check_bound == True)



class FMoWSentinel2(Dataset):
    def __init__(self, is_train = True, folder_path = '/mnt/h/era_reanalysis/', variables = None, transform = None, enc_transform = None, resolution = None, integer_values = None):
        """
        Args:
            folder_path (string): path to nc file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.is_train = is_train
        self.folder_path = folder_path
        self.variables = variables
        self.data_channels = len(variables)
        self.integer_values = integer_values
        self.resolution = resolution
        # Transforms
        self.transform = transform
        self.enc_transform = enc_transform
        
        # open the netcdf file and standardize
        self.file_paths = glob.glob(folder_path + '*01_hr.nc')
        self.hr_ds = xr.open_mfdataset(self.file_paths, engine = 'h5netcdf', parallel = True, chunks='auto')[variables].isel(level = 1)
        self.lr_ds = self.hr_ds.coarsen(latitude=10, longitude=10, boundary='trim').mean()
        # the using the last year as testset
        if self.is_train is False:
            self.last_year = max(np.unique(self.lr_ds.time.dt.year))
            self.hr_ds = self.hr_ds.sel(time=self.hr_ds.time.dt.year.isin([self.last_year]))
            self.lr_ds = self.lr_ds.sel(time=self.lr_ds.time.dt.year.isin([self.last_year]))
            
        self.hourly_grouped_mean = xr.open_dataset('/mnt/h/era_reanalysis/2016-2022_01_hourly_mean.nc', engine='h5netcdf').isel(level = 1)
        self.hourly_grouped_stds = xr.open_dataset('/mnt/h/era_reanalysis/2016-2022_01_hourly_stds.nc', engine='h5netcdf').isel(level = 1)

        self.hr_ds_standardized_hourly = ((self.hr_ds.groupby('time.hour') - self.hourly_grouped_mean).groupby('time.hour') / self.hourly_grouped_stds).compute()
        self.lr_ds_standardized_hourly = ((self.lr_ds.groupby('time.hour') - self.hourly_grouped_mean).groupby('time.hour') / self.hourly_grouped_stds).compute()
        
        # Calculate len
        self.time = self.hr_ds_standardized_hourly.time
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
        elif index < 6:
            hr2_t_idx = torch.randint(0, index, (1,))[0]
        else:
            hr2_t_idx = torch.randint(index - 6, index, (1,))[0]
        
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
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC , antialias = False),
            #transforms.RandomCrop(128),
            #transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC , antialias = False),
            #transforms.RandomCrop(128),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    

    selected_vars = ['U','V']
    is_train = False
    dataset = FMoWSentinel2(is_train, folder_path = '/mnt/h/era_reanalysis/', variables = selected_vars, transform = transform,enc_transform=enc_transform, 
                            resolution = 256, integer_values=False)
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True, generator=g) 

    # Display image and label.
    high, low, high2, high_norm_param, high2_norm_param  = next(iter(train_dataloader))
    print(f"target_hr: {high.size()}")
    print(f"target_lr: {low.size()}")
    print(f"hr_t': {high2.size()}")
    
    target_hr_sample = high[0,0:len(selected_vars),:,:].squeeze() #second vector change with number of data channels
    target_lr_sample = low[0].squeeze()
    hr_t_prime_sample = high2[0].squeeze()
    
    # calculate the L1 loss
    l1 = nn.L1Loss(reduction='mean')
    l1_loss = l1(target_lr_sample, target_hr_sample).item()
    # calculate psnr
    psnr = PeakSignalNoiseRatio()
    psnr_score = psnr(target_lr_sample, target_hr_sample).item()
    
    image = save_single_channel_visual(low, int(8 ** 0.5), 0, l1_loss, psnr_score)
    #plt.imsave('test.png', image)
    #test inverse scaling back to the original magnitude
    # unnorm_low = low * high_norm_param[:,:,1].view(-1, len(selected_vars), 1, 1) + high_norm_param[:,:,0].view(-1, len(selected_vars), 1, 1)
    # unnorm_high = high[:,0:len(selected_vars),:,:] * high_norm_param[:,:,1].view(-1, len(selected_vars), 1, 1) + high_norm_param[:,:,0].view(-1, len(selected_vars), 1, 1)
    # unnorm_high2 = high2 * high2_norm_param[:,:,1].view(-1, len(selected_vars), 1, 1) + high2_norm_param[:,:,0].view(-1, len(selected_vars), 1, 1)
    
    #normalized
    plt.imshow(hr_t_prime_sample[1,:,:], cmap=plt.cm.coolwarm, vmin = -4, vmax = 4)
    plt.imshow(target_hr_sample[1,:,:], cmap=plt.cm.coolwarm, vmin = -4, vmax = 4)
    plt.imshow(target_lr_sample[1,:,:], cmap=plt.cm.coolwarm, vmin = -4, vmax = 4)
    
    #unnormalized
    # plt.imshow(unnorm_high2[0,1,:,:], cmap=plt.cm.coolwarm, vmin = -40, vmax = 40)
    # plt.imshow(unnorm_high[0,1,:,:], cmap=plt.cm.coolwarm, vmin = -40, vmax = 40)
    # plt.imshow(unnorm_low[0,1,:,:], cmap=plt.cm.coolwarm, vmin = -40, vmax = 40)
    
    #plot on earth
    # import cartopy
    # import cartopy.crs as ccrs
    # import cartopy.feature as cfeature
    # def create_map(image):

    #     res = '10m'
    #     proj = ccrs.LambertConformal(central_longitude=-95, central_latitude=45)
    #     img = plt.imread(image)
    #     img_extent = (2.0715, 15.72, 46.9526, 54.5877)

    #     ax = plt.axes(projection = proj)
    #     # EXPLICIT CRS HERE:
    #     ax.set_extent([3.0889, 17.1128, 46.1827, 55.5482], crs=ccrs.PlateCarree())    

    #     land_10m = cfeature.NaturalEarthFeature('physical', 'land', res,
    #                                             edgecolor = 'face', 
    #                                             facecolor=cfeature.COLORS['land'],
    #                                             zorder=0)

    #     state_provinces_10m = cfeature.NaturalEarthFeature(category = 'cultural', 
    #                                                     name = 'admin_1_states_provinces_lines',
    #                                                     scale = res,
    #                                                     facecolor = none)

    #     ax.add_feature(state_provinces_10m, edgecolor='gray')
    #     ax.add_feature(land_10m)
    #     ax.add_feature(cartopy.feature.BORDERS.with_scale(res), linestyle='-', linewith=1)
    #     ax.add_feature(cartopy.feature.COASTLINE.with_scale(res), linestyle='-')

    #     # USE CORRECT CRS HERE
    #     plt.imshow(img, origin='upper', extent=img_extent, transform=ccrs.PlateCarree())

    #     plt.show()
    


    
    # transformation = transforms.ToPILImage()
    # img = transformation(target_lr_sample[1:2,:,:])
    # img.show()
    