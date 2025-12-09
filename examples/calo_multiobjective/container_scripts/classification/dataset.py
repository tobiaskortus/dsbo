"""
Dataset for the Classification model. Based on pytorch
"""
import os
import sys
import random
from typing import List, Optional, Tuple
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib
import matplotlib.pyplot as plt


class FlipAugmentation:
    def __call__(self,imgs):
        axis = random.randint(0,2)
        if axis == 0:
            return imgs
        else:
            return [
                np.flip(img, axis=axis)
                for img in imgs
            ]

class RotationAugmentation:
    def __call__(self,imgs):
        k = random.randint(0,3)
        return [
            np.rot90(img, k=k, axes=(1,2))
            for img in imgs
        ]

class ShiftAugmentation:
    def __call__(self,imgs):
        dx = random.randint(-3,3)
        dy = random.randint(-3,3)
        return self.shift_images(imgs,dx,dy)

    @staticmethod
    def shift_images(imgs,dx,dy):
        imgs = [
            np.roll(img, shift=dy, axis=1)
            for img in imgs
        ]
        imgs = [
            np.roll(img, shift=dx, axis=2)
            for img in imgs
        ]
        if dy > 0:
            for i in range(len(imgs)):
                imgs[i][:,:dy,:] = 0.
        if dy < 0:
            for i in range(len(imgs)):
                imgs[i][:,dy:,:] = 0.
        if dx > 0:
            for i in range(len(imgs)):
                imgs[i][:,:,:dx] = 0.
        if dx < 0:
            for i in range(len(imgs)):
                imgs[i][:,:,dx:] = 0.
        return imgs


class ClassificationDataset(Dataset):
    def __init__(
        self,
        classes: Tuple[str],
        input_data: pd.DataFrame,
        means: Optional[List[np.float32]] = None,
        stds: Optional[List[np.float32]] = None,
        augmentations: List = [],
    ):
        self.input_data = input_data
        self.input_data = self.filter_infs_and_nans(self.input_data)
        self.input_data = self.filter_empty_events(self.input_data)

        self.targets = np.concatenate(
            [
                self.input_data['Classes'][key].reshape(-1,1)
                for key in classes
            ],
            axis = 1,
        ).astype(np.float32)

        # Augmentations #
        self.augmentations = augmentations

        # Crop params #
        self.crop_size = 25

        # Get inputs #
        ls  = self.input_data['Inputs']['sensor_layer'].astype(np.float32)
        xs  = self.input_data['Inputs']['sensor_x'].astype(np.float32)
        ys  = self.input_data['Inputs']['sensor_y'].astype(np.float32)
        zs  = self.input_data['Inputs']['sensor_z'].astype(np.float32)
        dxs = self.input_data['Inputs']['sensor_dx'].astype(np.float32)
        dys = self.input_data['Inputs']['sensor_dy'].astype(np.float32)
        dzs = self.input_data['Inputs']['sensor_dz'].astype(np.float32)
        Es  = self.input_data['Inputs']['sensor_energy'].astype(np.float32)

        # True #
        self.particle_energy = self.input_data['Targets']['true_energy']

        # Make images #
        self.E_imgs = []
        self.x_imgs = []
        self.y_imgs = []
        for idx in np.unique(ls):
            mask = ls==idx
            assert np.all(mask == mask[0])
            mask = mask[0]
            E = Es[:,mask]
            x = xs[:,mask]
            y = ys[:,mask]
            z = zs[:,mask][:,:1]
            # should be the same for each x,y
            dx = dxs[:,mask][:,:1]
            dy = dys[:,mask][:,:1]
            dz = dzs[:,mask][:,:1]
            assert all(dx>0)
            assert all(dy>0)
            assert all(dz>0)
            # Add tot energy as context #

            if E.shape[-1] > 1:
                # granular image #
                width = int(math.sqrt(E.shape[-1]))
                assert width*width == E.shape[-1]
                E = E.reshape(-1,width,width)
                x = x.reshape(-1,width,width)
                y = y.reshape(-1,width,width)

            self.E_imgs.append(E)
            self.x_imgs.append(x)
            self.y_imgs.append(y)

        # Center and crop images #
        self.centroids = self.shower_centroids(self.E_imgs,self.x_imgs,self.y_imgs)
        E_imgs_cropped = [
            self.crop_and_center_images(
                imgs = Es,
                xs = xs,
                ys = ys,
                centers = self.centroids,
                size = self.crop_size,
            ).astype(np.float32)
            if Es.ndim == 3 else Es
            for Es,xs,ys in zip(self.E_imgs,self.x_imgs,self.y_imgs)
        ]

        # Make inputs #
        self.inputs = []
        for E_img in E_imgs_cropped:
            if E_img.ndim == 3:
                # normalise and scale
                E_img = E_img / (E_img.sum(axis=(1,2),keepdims=True) + 1e-9)
                E_img = np.expand_dims(E_img,axis=1)
                self.inputs.append(
                    np.concatenate(
                        [
                            E_img ** 0.25,
                            (E_img>0).astype(np.float32), # binary mask for empty cells
                        ],
                        axis = 1,
                    ),
                )
            else:
                self.inputs.append(np.log(1+E_img))

        # Make means and stds #
        if means is None:
            self.means = [
                [
                    np.array(
                        [
                            inputs[:,0][inputs[:,0] > 0].mean(), # energy
                            0., # bit mask
                        ],
                        dtype = np.float32,
                    )
                    if inputs.ndim==4
                    else inputs.mean(axis=0)
                    for inputs in self.inputs
                ]
            ]
        else:
            self.means = means
        if stds is None:
            self.stds = [
                [
                    np.array(
                        [
                            inputs[:,0][inputs[:,0] > 0].std() + 1e-10, # energy
                            1., # bit mask
                        ],
                        dtype = np.float32,
                    )
                    if inputs.ndim==4
                    else inputs.mean(axis=0)
                    for inputs in self.inputs
                ]
            ]
        else:
            self.stds = stds

        self.shape = (
            [inputs.shape[1:] for inputs in self.inputs],
            self.targets.shape[1]
        )

#        dev = "cuda" if torch.cuda.is_available() else "cpu"
#        self.c_means = [torch.tensor(a).to(dev) for a in self.means]
#        self.c_stds = [torch.tensor(a).to(dev) for a in self.stds]

    @staticmethod
    def shower_centroids(imgs,xs,ys):
        imgx_total = np.zeros(imgs[0].shape[0])
        imgy_total = np.zeros(imgs[0].shape[0])
        img_total  = np.zeros(imgs[0].shape[0])
        for img,x,y in zip(imgs,xs,ys):
            if img.ndim == 3:
                imgx_total += (img * x).sum(axis=(1,2))
                imgy_total += (img * y).sum(axis=(1,2))
                img_total  += img.sum(axis=(1,2))
        return imgx_total / (img_total+1e-9), imgy_total / (img_total+1e-9)

    @staticmethod
    def layer_centroid(centroid,x,y):
        dist2 = (x-centroid[0])**2 + (y-centroid[1])**2
        y_pix,x_pix = np.unravel_index(np.argmin(dist2), dist2.shape)
        return x_pix,y_pix

    def crop_and_center_images(self,imgs,xs,ys,centers,size):
        cropped_imgs = np.zeros((imgs.shape[0],size,size))
        losses = np.zeros(imgs.shape[0])
        for i in range(imgs.shape[0]):
            center = self.layer_centroid(
                centroid = (centers[0][i],centers[1][i]),
                x = xs[i],
                y = ys[i],
            )
            cropped_imgs[i] = self.crop_and_center_image(
                img = imgs[i],
                center = center,
                size = size,
            )
            losses[i] = (imgs[i].sum()-cropped_imgs[i].sum()) / (imgs[i].sum()+1e-9)

        print (f'Crop {imgs.shape[1]}x{imgs.shape[2]} -> {size}x{size}: losses = {losses.mean()*100:5.3f}% +/- {losses.std()*100:5.3f}%')
        return cropped_imgs


    @staticmethod
    def crop_and_center_image(img, center, size):
        H, W = img.shape
        x, y = center
        x = int(round(x))
        y = int(round(y))

        half = size // 2

        # Compute desired crop boundaries
        ymin = y - half
        ymax = y + half
        xmin = x - half
        xmax = x + half

        # Create an empty (zero-padded) crop
        crop = np.zeros((size, size), dtype=img.dtype)

        # Find overlapping region between crop and image
        y0_img = max(ymin, 0)
        y1_img = min(ymax, H)
        x0_img = max(xmin, 0)
        x1_img = min(xmax, W)

        # Corresponding coordinates in the crop
        y0_crop = y0_img - ymin
        y1_crop = y0_crop + (y1_img - y0_img)
        x0_crop = x0_img - xmin
        x1_crop = x0_crop + (x1_img - x0_img)

        # Copy actual pixels into padded crop
        crop[y0_crop:y1_crop, x0_crop:x1_crop] = img[y0_img:y1_img, x0_img:x1_img]

        return crop


    def filter_infs_and_nans(self, data):
        '''
        Removes all events that contain infs or nans.
        '''
        for superkey, submf in data.items():
            for key in submf.keys():
                data[superkey][key] = np.nan_to_num(submf[key],nan=0.,posinf=0.,neginf=0.)
        return data

    def filter_empty_events(self, data):
        idx = np.where(data['Inputs']['sensor_energy'].sum(axis=1)>0)[0]
        for superkey, submf in data.items():
            for key in submf.keys():
                data[superkey][key] = submf[key][idx]
        return data

#    def unnormalize_detector(self, detector: torch.Tensor):
#        return detector * self.c_stds[1] + self.c_means[1]
#
#    def normalize_detector(self, detector: torch.Tensor):
#        return (detector - self.c_means[1]) / self.c_stds[1]

    def __len__(self) -> int:
        return len(self.targets)

    def normalize_img(self,img,means,stds):
        assert img.ndim == 3
        out = (img- means.reshape(-1,1,1)) / stds.reshape(-1,1,1)
        return out

    def __getitem__(self, idx: int):
        imgs = [inputs[idx] for inputs in self.inputs]

        for augmentaton in self.augmentations:
            imgs = augmentaton(imgs)

        imgs = [
            self.normalize_img(img,self.means[0][i],self.stds[0][i])
            for i,img in enumerate(imgs)
        ]

        return imgs, self.targets[idx]

    def plot(self, idx, size=None,figname=None):
        E_imgs = [
            E_img[idx]
            for E_img in self.E_imgs
        ]
        x_imgs = [
            x_img[idx]
            for x_img in self.x_imgs
        ]
        y_imgs = [
            y_img[idx]
            for y_img in self.y_imgs
        ]
        tot_E = sum([E_img.sum() for E_img in E_imgs])
        if tot_E == 0:
            print ('No measured energy')
            return
        E_min = min([E_img[E_img>0].min() for E_img in E_imgs if E_img.sum()>0])
        E_max = max([E_img[E_img>0].max() for E_img in E_imgs if E_img.sum()>0])

        if size is not None:
            centroid = self.shower_centroids(
                imgs = [E_img[None,:,:] for E_img in E_imgs],
                xs = [x_img[None,:,:] for x_img in x_imgs],
                ys = [y_img[None,:,:] for y_img in y_imgs],
            )
            E_imgs_cropped = [
                self.crop_and_center_image(
                    img = Es,
                    center = self.layer_centroid(
                        centroid = centroid,
                        x = xs,
                        y = ys,
                    ),
                    size = size,
                )
                if Es.ndim == 2 else Es
                for Es,xs,ys in zip(E_imgs,x_imgs,y_imgs)
            ]
            x_imgs_cropped = [
                self.crop_and_center_image(
                    img = xs,
                    center = self.layer_centroid(
                        centroid = centroid,
                        x = xs,
                        y = ys,
                    ),
                    size = size,
                )
                if xs.ndim == 2 else xs
                for xs,ys in zip(x_imgs,y_imgs)
            ]
            y_imgs_cropped = [
                self.crop_and_center_image(
                    img = ys,
                    center = self.layer_centroid(
                        centroid = centroid,
                        x = xs,
                        y = ys,
                    ),
                    size = size,
                )
                if ys.ndim == 2 else ys
                for xs,ys in zip(x_imgs,y_imgs)
            ]
            E_imgs = E_imgs_cropped
            x_imgs = x_imgs_cropped
            y_imgs = y_imgs_cropped

        N = len(E_imgs)
        fig,axs = plt.subplots(ncols=N,figsize=(6*N,5))
        plt.suptitle(f'Target = {self.targets[idx]} [true energy = {self.particle_energy[idx]:.3f} GeV]',fontsize=18)
        plt.subplots_adjust(left=0.05,right=0.95,wspace=0.3)
        if not isinstance(axs,np.ndarray):
            axs = np.array([axs])
        for i in range(N):
            E = E_imgs[i]
            x = x_imgs[i]
            y = y_imgs[i]
            dx = x[int(x.shape[0]//2),int(x.shape[1])//2+1] - x[int(x.shape[0]//2),int(x.shape[1])//2]
            dy = y[int(y.shape[0]//2)+1,int(y.shape[1])//2] - y[int(y.shape[0]//2),int(y.shape[1])//2]
            assert dx > 0
            assert dy > 0
            if E.ndim != 2:
                E = np.expand_dims(E,axis=1)
            im = axs[i].imshow(
                E,
                extent = [x.min()-dx,x.max()+dx,y.min()-dy,y.max()+dy],
                norm = matplotlib.colors.LogNorm(vmin=E_min,vmax=E_max),
                origin='lower',
            )
            axs[i].set_xlabel('x [mm]',fontsize=16)
            axs[i].set_ylabel('y [mm]',fontsize=16)
            cbar = fig.colorbar(im, ax=axs[i], shrink=0.9)
            cbar.set_label('Energy deposit [MeV]',fontsize=16)
            axs[i].set_title(f'Layer {i}')
        plt.show()
        if figname is not None:
            fig.savefig(figname)
            print (f'Saved as {figname}')


if __name__ == '__main__':
    import pathlib
    sys.path.append(os.path.abspath(pathlib.Path(__file__).parent.parent))
    from merge import SuperMiniFrame
    dataset = ClassificationDataset(SuperMiniFrame.read_pickle(sys.argv[1]))

#    fig,axs = plt.subplots(ncols=len(dataset.inputs),figsize=(5*len(dataset.inputs),4))
#    for i in range(len(dataset.inputs)):
#        imgs = dataset.inputs[i][:,0,:,:]
#        e_imgs = imgs[(dataset.targets==1).ravel()]
#        g_imgs = imgs[(dataset.targets==0).ravel()]
#        e_frac = (e_imgs>0).sum(axis=(1,2)) / (e_imgs.shape[1]*e_imgs.shape[2])
#        g_frac = (g_imgs>0).sum(axis=(1,2)) / (g_imgs.shape[1]*g_imgs.shape[2])
#        bins = np.linspace(0,max([e_frac.max(),g_frac.max()]),100)
#        axs[i].hist(
#            e_frac,
#            bins = bins,
#            color = 'blue',
#            histtype = 'step',
#            label = 'e'
#        )
#        axs[i].hist(
#            g_frac,
#            bins = bins,
#            color = 'red',
#            histtype = 'step',
#            label = r'$\gamma$'
#        )
#        axs[i].set_yscale('log')
#        axs[i].set_xlabel('Fraction of non-empty cells')
#        axs[i].set_title(f'Layer {i}')
#        axs[i].legend()
#    fig.savefig('e_fraction.png')
#    plt.show()
#

    from IPython import embed; embed()
