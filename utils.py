"""Helper functions."""

import re
import random
from typing import Optional, Union, Tuple, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
from pathlib import Path


def parallel_map(iterables: Union[list, iter], function: callable, n_jobs: Optional[int] = 16,
                 use_kwargs: Optional[bool] = False, front_num: Optional[int] = 3,
                 show_progress_bar: Optional[bool] = True, initial_value: Optional[list] = None,
                 raise_errors: Optional[bool] = False, include_errors: Optional[bool] = True,
                 extend_result: Optional[bool] = False, return_output: Optional[bool] = True,
                 add_func: Optional[callable] = None) -> Union[list, None]:
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
        extend_result: Whether the resultant list should be extended rather than appended to.
        return_output: Whether to return a list containing the output values of the function.
            If False, this function does not return None.
        add_func: A custom function for adding the output values of the function to the result list.
            This function has two parameters, the value to add and the list to add it to, and it
            should mutate the list.

    Preconditions:
        - n_jobs >= 1
        - front_num >= 0
    """
    if isinstance(iterables, list):
        front = [function(**a) if use_kwargs else function(a) for a in iterables[:front_num]]
        iterables = iterables[front_num:]
    else:
        front = []
        for _ in range(front_num):
            a = next(iterables)
            front.append(function(**a) if use_kwargs else function(a))

    def _add_func(x: object, output: list) -> None:
        """Add a value to the output list."""
        # No reason to add if we aren't returning the output.
        if not return_output:
            return

        if add_func is not None:
            add_func(x, output)
        else:
            if extend_result:
                output.extend(x)
            else:
                output.append(x)

    output = initial_value or list()
    for x in front:
        _add_func(x, output)

    # If n_jobs == 1, then we are not parallelising, run all elements serially.
    if n_jobs == 1:
        for a in tqdm(iterables):
            x = function(**a) if use_kwargs else function(a)
            _add_func(x, output)

        return output if return_output else None

    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = [
            pool.submit(function, **a) if use_kwargs else
            pool.submit(function, a) for a in iterables
        ]

        for _ in tqdm(as_completed(futures), total=len(futures), unit='it',
                      unit_scale=True, disable=not show_progress_bar):
            # Do nothing...This for loop is just here to iterate through the futures
            pass

    # Don't bother retrieving the results from the future...If we don't return anything.
    if not return_output:
        return None

    for _, future in tqdm(enumerate(futures)):
        try:
            _add_func(future.result(), output)
        except Exception as exception:
            if raise_errors:
                raise exception
            if include_errors:
                _add_func(exception, output)

    return output


def set_seed(seed: int) -> None:
    """Sets the seed in random, numpy, and tensorflow.

    Args:
        seed: The seed of the random engine.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Set TensorFlow seed
    import tensorflow as tf
    tf.random.set_seed(seed)


def get_next_run_id(directory: Union[str, Path], run_name: str,
                    initial_id: Optional[int] = 1, padding: Optional[int] = 5) -> str:
    """
    Return the next id of the run (i.e. what to name the output folder).

    This looks for all folders in the specified directory whose name,
    or run id, is in the format "{id}-{run_name}". The next run id
    prefix is one plus the maximum of all ids with the specified run_name.

    Args:
        directory: The directory containing all the runs.
        run_name: The name of the run. This should be unique across different runs.
        initial_id: The id to start with, if no runs exist.
        padding: The amount to pad the numerical id.
    """
    directory = Path(directory)
    pattern = r'(\d*)-{}'.format(run_name)
    next_id = initial_id
    for path in directory.iterdir():
        if not path.is_dir(): continue

        m = re.match(pattern, path.name)
        if not m: continue
        next_id = max(int(m.group(1)) + 1, next_id)

    return '{}-{}'.format(str(next_id).zfill(padding), run_name)


def rgb_lerp(colour_a: Tuple[int, int, int], colour_b: Tuple[int, int, int], t: float) \
    -> Tuple[int, int, int]:
    """Linearlly interpolates from one colour to another.

    Preconditions:
        - all(0 <= x <= 255 for x in colour_a)
        - all(0 <= x <= 255 for x in colour_b)
        - 0 <= t <= 1
    """
    r1, g1, b1 = colour_a
    r2, g2, b2 = colour_b
    return (
        int((r2 - r1) * t + r1),
        int((g2 - g1) * t + g1),
        int((b2 - b1) * t + b1),
    )


def rgb_to_str(colour: Tuple[int, int, int]) -> str:
    """Return a string representation of the colour tuple.

    Preconditions:
        - all(0 <= x <= 255 for x in colour)
    """
    return 'rgb({})'.format(','.join(str(x) for x in colour))


def list_join(lst: List[Any], conjuction: Optional[str] = 'and',
              format_func: Optional[callable] = str,
              oxford_comma: Optional[bool] = True) -> str:
    """Join a list into a human-readable English sentence.

    Args:
        lst: The list to join.
        conjuction: Conjunction word used on the last two items.
        format_func: A function that takes in a string as input and returns
            a formatted string.
        oxford_comma: Whether to use an oxford comma.

    >>> list_join(['David', 'Mario', 'Shon'])
    David, Mario, and Shon
    """
    if not lst:
        return ''
    if len(lst) <= 2:
        return f' {conjuction} '.join(lst)

    comma = ',' if oxford_comma else ''
    return ', '.join(lst[:-1]) + f'{comma} {conjuction} ' + lst[-1]


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'extra-imports': [
            'random',
            'typing',
            'concurrent.futures',
            'tqdm',
            'numpy'
            'tensorflow',
        ],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })
