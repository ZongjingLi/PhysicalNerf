import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from utils import *

class PACDataset(Dataset):
    def __init__(self,object_name,single_timestep = True,resolution = (64,64)):
        root_dir = "/Users/melkor/Documents/datasets/pac_data/{}/".format(object_name)
        self.root_dir = root_dir
        self.resolution = resolution

        self.scene_data = load_json(self.root_dir + "all_data.json")

        self.single_timestep = single_timestep

        self.data = {}

        for item in self.scene_data:
            
            file_dir = self.root_dir + item["file_path"][2:]
            image = torch.tensor(np.array(Image.open(file_dir).resize(self.resolution))).float()
            camera2world = item["c2w"]
            time = item["time"]

            if item["file_path"][10] == "_":view_id = item["file_path"][9]
            else:view_id = item["file_path"][9:11]
            if view_id in self.data:
                self.data[view_id].append({"image":image,"time":time,"camera2world":torch.tensor(camera2world)})
            else:
                self.data[view_id] = [{"image":image,"time":time,"camera2world":torch.tensor(camera2world)}]
        def time_order(data):return data["time"]
        for name in self.data:
            self.data[name].sort(key = time_order)
            #print(name,len(self.data[name]))

    def __getitem__(self, idx):
        if self.single_timestep:
            K_frame = 1
            images = []
            cameras = []
            for view_name in self.data:
                view_data = self.data[view_name]
                images.append(view_data[K_frame]["image"])
                cameras.append(view_data[K_frame]["camera2world"])
            images = torch.stack(images)
            cameras = torch.stack(cameras)
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            B = cameras.shape[0]
            last = torch.zeros([B,1,4])
            last[:,:,3] = 1

            return {"images":images, "c2w":torch.cat([cameras,last],dim = 1)}
        else:
            return 0

    def __len__(self):
        if self.single_timestep:
            return 1
        else:
            return len(self.data["0"])


class PACDynamicDataset(Dataset):
    def __init__(self,object_name,single_timestep = True,resolution = (64,64)):
        root_dir = "/Users/melkor/Documents/datasets/pac_data/{}/".format(object_name)
        self.root_dir = root_dir
        self.resolution = resolution

        self.scene_data = load_json(self.root_dir + "all_data.json")

        self.single_timestep = single_timestep

        self.data = {}

        for item in self.scene_data:
            
            file_dir = self.root_dir + item["file_path"][2:]
            image = torch.tensor(np.array(Image.open(file_dir).resize(self.resolution))).float()
            camera2world = item["c2w"]
            time = item["time"]

            if item["file_path"][10] == "_":view_id = item["file_path"][9]
            else:view_id = item["file_path"][9:11]
            if view_id in self.data:
                self.data[view_id].append({"image":image,"time":time,"camera2world":torch.tensor(camera2world)})
            else:
                self.data[view_id] = [{"image":image,"time":time,"camera2world":torch.tensor(camera2world)}]
        def time_order(data):return data["time"]
        for name in self.data:
            self.data[name].sort(key = time_order)
            #print(name,len(self.data[name]))

    def __getitem__(self, idx):
        if self.single_timestep:
            K_frame = idx
            images = []
            cameras = []
            for view_name in self.data:
                view_data = self.data[view_name]
                images.append(view_data[K_frame]["image"])
                cameras.append(view_data[K_frame]["camera2world"])
            images = torch.stack(images)
            cameras = torch.stack(cameras)
            return {"images":images, "c2w":cameras}
        else:
            return 0

    def __len__(self):

        return len(self.data["0"])