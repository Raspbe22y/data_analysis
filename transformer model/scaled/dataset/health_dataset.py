import pandas as pd
from torch.utils.data import Dataset
import os
import torch
import numpy as np
class HealthDataset(Dataset):
    def __init__(self, output_csv, input_csv):
        df = pd.read_csv(output_csv)
        self.output = df
        self.input_csv = input_csv
        self.input_list = sorted(os.listdir(input_csv))
        self.check_input_list()

    def check_input_list(self):
        new_input_list = []
        for name in self.input_list:
            patient_id, season = name.split(".")[0].split("_")
            if ((self.output['person'].astype(str) == patient_id) &
                    (self.output['season'] == season)).any():
                new_input_list.append(name)

        self.input_list = new_input_list

    def __len__(self):
        return len(self.input_list)

    def get_input(self, i):
        name = self.input_list[i]
        # data = pd.read_excel(os.path.join(self.input_csv, name))
        # data = data.values[:300]
        data = np.load(os.path.join(self.input_csv, name.replace(".xlsx", ".npy")))
        data[:, 0] = data[:, 0] / 15    # br_avg
        data[:, 1] = data[:, 1] / 3     # br_std
        data[:, 2] = data[:, 2] / 1     # act_level
        data[:, 3] = data[:, 3] / 55    # step_count
        data[:, 4] = data[:, 4] / 800   # latitude
        data[:, 5] = data[:, 5] / 0.2   # longitude
        data[:, 6] = data[:, 6] / 1     # pm_dose_hr
        data[:, 7] = data[:, 7] / 0.1   # pm_dose_rate
        data[:, 8] = data[:, 8] / 8     # pm2_5
        data[:, 9] = data[:, 9] / 30    # temperature
        data[:, 10] = data[:, 10] / 50    # humidity
        # data[:, 0] = data[:, 0] / 15  # convert milliseconds to seconds
        # data[:, 1] = data[:, 1] / 3  # normalize heart rate
        # data[:, 2] = data[:, 2] / 1  # normalize skin temperature
        # data[:, 3] = data[:, 3] / 800  # normalize galvanic skin response
        # data[:, 4] = data[:, 4] / 55  # normalize step count
        # data[:, 5] = data[:, 5] / 0.2  # normalize calories
        # data[:, 6] = data[:, 6] / 1  # normalize activity intensity
        # data[:, 7] = data[:, 7] / 0.1  # normalize metabolic equivalent
        # data[:, 8] = data[:, 8] / 8  # normalize ambient light
        # data[:, 9] = data[:, 9] / 30  # normalize skin humidity
        # data[:, 10] = data[:, 10] / 50  # normalize UV exposure
        return torch.tensor(data, dtype=torch.float32)

    def __getitem__(self, idx):
        name = self.input_list[idx]
        patient_id, season = name.split(".")[0].split("_")

        x = self.get_input(idx)
        x = x.detach().clone().to(dtype=torch.float32)
        y_df = self.output[
            (self.output['person'].astype(str) == patient_id) &
            (self.output['season'] == season)
        ].values[:, 2:].astype(np.float32)
        # print(y_df.shape)
        y = torch.tensor(y_df[0], dtype=torch.float32)
        y[2] = y[2] / 1000  # convert percentage to decimal
        y[-1] = y[-1] / 1000  # convert percentage to decimal
        return x, y
