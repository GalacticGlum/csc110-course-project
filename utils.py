"""Helper functions."""

from concurrent.futures import (
    ThreadPoolExecutor,
    ProcessPoolExecutor,
    as_completed
)

from tqdm import tqdm


def parallel_map(array: list, function: callable, n_jobs: int = 16, use_kwargs: bool = False,
                 front_num: int = 3, multithread: bool = False, show_progress_bar: bool = True,
                 extend_result: bool = False, initial_value: list = None):
    """
    A parallel version of the map function with a progress bar.
    :note:
        This is a utility function for running parallel jobs with progress
        bar. Originally from http://danshiebler.com/2016-09-14-parallel-progress-bar/.
        The implementation is identical to the source; however, the documentation and
        code style has been modified to fit the style of this codebase.
    :param array:
        An array to iterate over.
    :param function:
        A python function to apply to the elements of array
    :param n_jobs:
        The number of cores to use. Defaults to 16.
    :param use_kwargs:
        Whether to consider the elements of array as dictionaries of
        keyword arguments to function. Defaults to ``False``.
    :param front_num:
        The number of iterations to run serially before kicking off the
        parallel job. Useful for catching bugs
    :param multithread:
        If ``True``, a :class:``concurrent.futures.ThreadPoolExecutor`` will be used rather
        than a :class:``concurrent.futures.ProcessPoolExecutor``. Defaults to ``False``.
    :param show_progress_bar:
        Indicates whether a loading progress bar should be displayed while the process runs.
        Defaults to ``True``.
    :param extend_result:
        Indicates whether the resultant list should be extended rather than appended to.
        Defaults to ``False``. Note that this requires that the return value of ``function``
        is an array-like object.
    :param initial_value:
        The initial value of the resultant array. This should be an array-like object.
        Defaults to None.
    :returns:
        A list of the form [function(array[0]), function(array[1]), ...].
    """
    # We run the first few iterations serially to catch bugs
    front = []
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]

    # If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs == 1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]

    # Assemble the workers
    pool_type = ThreadPoolExecutor if multithread else ProcessPoolExecutor
    with pool_type(max_workers=n_jobs) as pool:
        # Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]

        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True,
            'disable': not show_progress_bar
        }

        # Print out the progress as tasks complete
        for _ in tqdm(as_completed(futures), **kwargs):
            pass

    out = initial_value or list()
    out.extend(front)

    # Get the results from the futures.
    _add_func = lambda x: out.extend(x) if extend_result else out.append(x)
    for _, future in tqdm(enumerate(futures)):
        try:
            _add_func(future.result())
        except Exception as e:
            _add_func(e)

    return out


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'extra-imports': [
            'tqdm',
            'concurrent.futures'
        ],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
