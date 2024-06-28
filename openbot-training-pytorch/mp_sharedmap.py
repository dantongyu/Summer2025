import collections
import os

import torch
import torch.multiprocessing as mp
import numpy as np

global_data = {}


def init_worker(i, output_type_, output_):
    global input, fn, output, output_type
    output_type = output_type_
    output = output_
    input = global_data[i][0]
    fn = global_data[i][1]
    # XXX: https://github.com/pytorch/pytorch/issues/17199#issuecomment-493672631
    # XXX: or using intel openmp: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#intel-openmp-runtime-library-libiomp
    # XXX: intel openmp is not suggested because it has CPU affinity issue
    torch.set_num_threads(1)
    os.sched_setaffinity(0, range(os.cpu_count())) 


def process_data_wrapper(i):
    result = fn(input[i])

    if output_type == "list":
        for j in range(len(output)):
            output[j][i] = result[j]
    else:
        output[i] = result


def initialize_output(size, output_t):
    if isinstance(output_t, np.ndarray):
        shape = (size, *output_t.shape)
        dtype = getattr(torch, output_t.dtype.name)
        size = size * output_t.nbytes
    elif isinstance(output_t, torch.Tensor):
        shape = (size, *output_t.shape)
        dtype = output_t.dtype
        size = size * output_t.untyped_storage().nbytes()
    elif isinstance(output_t, int):
        shape = (size, )
        dtype = torch.int32
        size = size * 4
    elif isinstance(output_t, float):
        shape = (size, )
        dtype = torch.float32
        size = size * 4
    elif isinstance(output_t, bool):
        shape = (size, )
        dtype = torch.bool
        size = size
    else:
        raise Exception(f"Cannot handle type {type(output_t)} for output")

    data = torch.UntypedStorage._new_shared(np.prod(size))
    return torch.tensor([], dtype=dtype, device=data.device).set_(data).view(*shape)


def map(fn, input, num_workers=None):
    # Input has to be a iterable
    input = list(input)
    # Test the function output
    # Output has to be either a numpy/tensor array or primitive type (int, float, bool)
    # List or tuple are allowed if they contain only supported types
    output_t = fn(input[0])

    # Preallocate the output buffer in the main process
    if isinstance(output_t, collections.abc.Sequence):
        output_type = "list"
        output = []
        for i in output_t:
            output.append(initialize_output(len(input), i))
    else:
        output_type = "array"
        output = initialize_output(len(input), output_t)

    # Utilize the linux COW to share the data between processes
    # The input data is read only, so the memory is shared between processes
    while True:
        i = np.random.randint(0, 0xffffffff)
        if i not in global_data:
            global_data[i] = (input, fn)
            break

    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(ctx.cpu_count() if num_workers is None else num_workers, initializer=init_worker, initargs=(i, output_type, output)) as pool:
            pool.map(process_data_wrapper, list(range(len(input))))

        return output

    finally:
        del global_data[i]
