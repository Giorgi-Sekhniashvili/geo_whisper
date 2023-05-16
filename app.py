import os

import huggingface_hub
import lightning as L

from train import train

huggingface_hub.login(os.environ["HUGGINGFACE_ACCESS_TOKEN"])


class TrainingComponent(L.LightningWork):
    def run(self):
        train()


compute = L.CloudCompute("gpu", disk_size=50, idle_timeout=30)
component = TrainingComponent(cloud_compute=compute)
app = L.LightningApp(root=component)
