import torch
import torch.utils.data


class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    """
    DataLoader that do not re-init workers
    https://github.com/pytorch/pytorch/issues/15849
    """
    def __init__(self, dataset, *args, **kwargs):
        # this is only required when using GPU for preprocessing
        # kwargs["multiprocessing_context"] = "spawn"

        if hasattr(dataset, "dataset_weight"):
            kwargs["shuffle"] = False
            kwargs["sampler"] = torch.utils.data.WeightedRandomSampler(
                    dataset.dataset_weight, len(dataset.dataset_weight))

        super().__init__(dataset, *args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

    def close(self):
        # Release the processes resources
        self.iterator._shutdown_workers()
