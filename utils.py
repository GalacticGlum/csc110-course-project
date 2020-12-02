"""Helper functions."""

from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def parallel_map(iterables: list, function: callable, n_jobs: Optional[int] = 16,
                 use_kwargs: Optional[bool] = False, front_num: Optional[int] = 3,
                 show_progress_bar: Optional[bool] = True, initial_value: Optional[list] = None,
                 raise_errors: Optional[bool] = False, include_errors: Optional[bool] = True) \
        -> list:
    """A parallel version of the map function with a progress bar.
    Return a list of the form [function(iterables[0]), function(iterables[1]), ...].

    Args:
        iterables: A sequence, collection, or iterator object.
        function: A function to apply to the elements of `iterables`.
        n_jobs: The number of jobs to run.
        use_kwargs: Whether to consider the elements of iterables as dictionaries of
            keyword arguments to function.
        front_num: The number of iterations to run serially before dispatching the
            parallel jobs. Useful for catching exceptions.
        show_progress_bar: Whether to show a progress bar while the jobs run.
        initial_value: The initial value of the output list.
            This should be an iterables-like object.
        raise_errors: Whether to raise errors.
        include_errors: Whether to include the errors in the output list.

    Preconditions:
        - n_jobs >= 1
        - front_num >= 0
    """
    front = [function(**a) if use_kwargs else function(a) for a in iterables[:front_num]]
    # If n_jobs == 1, then we are not parallelising, run all elements serially.
    if n_jobs == 1:
        return front + [
            function(**a) if use_kwargs else function(a)
            for a in tqdm(iterables[front_num:])
        ]

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = [
            pool.submit(function, **a) if use_kwargs else
            pool.submit(function, a) for a in iterables[front_num:]
        ]

        for _ in tqdm(as_completed(futures), total=len(futures), unit='it',
                      unit_scale=True, disable=not show_progress_bar):
            # Do nothing...This for loop is just here to iterate through the futures
            pass

    output = initial_value or list()
    output.extend(front)

    for _, future in tqdm(enumerate(futures)):
        try:
            output.append(future.result())
        except Exception as exception:
            if raise_errors:
                raise exception
            if include_errors:
                output.append(exception)

    return output


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
