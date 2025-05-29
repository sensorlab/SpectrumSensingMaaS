import numpy as np
import torch
import os
from typing import Tuple, Union
from torch.utils import data
from torchvision import transforms as T
import math
import h5py

class BaseSignalDataset(object):
    # Global dataset constants
    RSS_MAX = 133
    RSS_MIN = 47

    # Should it go time-first or bandwith-first?
    __time_first__ = False

    def __init__(self, window:int, stride:int, dataset_path:Union[os.PathLike, str], verbose:bool=False) -> None:
        super().__init__()

        # Verbosity for debugging
        self.verbose = verbose

        # Preprocess window and stride size
        self.window = (window, window) if isinstance(window, int) else window
        self.stride = (stride, stride) if isinstance(stride, int) else stride

        # Sanity checks
        assert self.window[0] >= self.stride[0]
        assert self.window[1] >= self.stride[1]
        assert self.window[0] != 0 and self.window[1] != 0
        assert self.stride[0] != 0 and self.stride[1] != 0

        # Open pointer to HDF5 file
        self.filename = dataset_path
        with h5py.File(self.filename, mode="r", swmr=True) as fp:
            #self.fp = h5py.File(self.filename, mode="r", swmr=True)
            #Obtain max time range and max bandwidth range
            self.t_max, self.bw_max = fp["rss"].shape

        self.t_stride, self.bw_stride = self.stride
        self.t_window, self.bw_window = self.window

        self.t_steps = math.ceil(self.t_max / self.t_stride)
        self.bw_steps = math.ceil(self.bw_max / self.bw_stride)

        print()

        if self.verbose: print(f'Dataset samples (w.r.t. window and strides) T={self.t_steps:,} BW={self.bw_steps:,} TOTAL={self.t_steps * self.bw_steps:,}')

        if self.verbose:
            print('After correction', self.t_steps, self.bw_steps)
            print('ranges over BW')
            for idx in range(self.bw_steps):
                print('[', idx * self.bw_stride, ',', idx * self.bw_stride + self.bw_window - 1, ']')

            print(f'Overflow: {idx * self.bw_stride + self.bw_window - self.bw_max}')


    def __get_absolute_position__(self, idx) -> Tuple[int, int]:
        """Obtain absolute position in dataset (for debugging only)"""
        if self.__time_first__:
            t_step = idx % self.t_steps
            bw_step = idx // self.t_steps
        else:
            t_step = idx // self.bw_steps
            bw_step = idx % self.bw_steps

        return (t_step * self.t_stride, bw_step * self.bw_stride)


    def __len__(self) -> int:
        """Return the total number of available samples, with respect to window, stride, and limit"""
        total_samples = int(self.t_steps * self.bw_steps)

        if self.verbose:
            print(f'{self.t_steps} * {self.bw_steps} == {total_samples:,}')

        return total_samples


    def __getitem__(self, index: int) -> np.ndarray:
        if self.__time_first__:
            t_step = index % self.t_steps
            bw_step = index // self.t_steps
        else:
            t_step = index // self.bw_steps
            bw_step = index % self.bw_steps

        if self.verbose:
            print(f'idx[{index}] => Steps(t={t_step}, bw={bw_step}) => Abs(t={t_step*self.t_stride}, bw={bw_step*self.t_stride})')

        # Calculate range in time dimension
        #t_start, t_end = t_step * self.t_stride, (t_step + 1) * self.t_stride
        t_start = t_step * self.t_stride
        t_end = (t_start + self.window[0])
        
        # Calulate range in bandwith dimension
        #bw_start, bw_end = bw_step * self.bw_stride, (bw_step + 1) * self.bw_stride
        bw_start = bw_step * self.bw_stride
        bw_end = (bw_start + self.window[1])
        
        with h5py.File(self.filename, mode="r", swmr=True) as fp:
            # Obtain raw data
            raw_data = fp["rss"][t_start:t_end, bw_start:bw_end]

            if raw_data.shape != self.window:
                assert raw_data.shape != (0, 0)

                actual_t_size, actual_bw_size = raw_data.shape
                required_t_size, required_bw_size = self.window

                pad = (
                    (0, required_t_size - actual_t_size),
                    (0, required_bw_size - actual_bw_size),
                )

                raw_data = np.pad(raw_data, pad, mode="constant", constant_values=self.RSS_MAX)

            return raw_data
        
class SignalDatasetV2(data.Dataset):
    def __init__(self, window:int, stride:int, dataset_path:Union[os.PathLike, str], labels=[], limit=None, three_channels=False) -> None:
        super().__init__()
        self.dataset = BaseSignalDataset(window, stride, dataset_path, verbose=False)
        self.labels = labels
        self.three_channels = three_channels
        self.limit = limit
        assert not self.limit or self.limit < len(self.dataset)

    def __len__(self) -> int:
        return self.limit if self.limit else len(self.dataset)

    def __getitem__(self, index:int) -> np.ndarray:
        label = self.labels[index] if self.labels else 0
        image = self.dataset[index]
        three_channels = self.three_channels

        # Transform pictures

        # convert int array to float array
        image = image.astype(np.float32)

        # Invert the values (noise floor will be minimum)
        image *= -1
        _min, _max = -BaseSignalDataset.RSS_MAX, -BaseSignalDataset.RSS_MIN

        # min-max scaling
        image = (image - _min) / (_max - _min)

        # Add channel
        image = np.expand_dims(image, axis=0)

        # Convert to tensor
        image = torch.from_numpy(image)

        if three_channels:
            # Make it 3 channel image
            transform_3_channel = T.Compose([T.Resize((224, 224), antialias=True), T.ToPILImage(), 
                                             T.Grayscale(3), T.ToTensor()])
            
            image = transform_3_channel(image)
            # image =  T.functional.adjust_gamma(image, gamma=1/2)

        return image, label
