from transformers import TrainerCallback
from transformers.trainer_pt_utils import IterableDatasetShard
from torch.utils.data import IterableDataset


# trainer callback to reinitialise and reshuffle the streamable datasets at the beginning of each epoch
class ShuffleCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDatasetShard):
            pass  # set_epoch() is handled by the Trainer
        elif isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)
