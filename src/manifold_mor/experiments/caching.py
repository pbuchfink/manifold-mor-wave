'''Methods to cache experiment results on disk.'''
import os
import pickle
from functools import wraps
from typing import Any, Callable, TYPE_CHECKING

from manifold_mor.context import get_current_context

if TYPE_CHECKING:
    from manifold_mor.experiments.basic import ManifoldMorExperiment

PATH_CACHED_RESULTS = 'cached_results'

# caching operations
def cached_result(fun: Callable) -> Any:
    """Wraps a function to cache its result via pickle. May be globally toggled with
        context.options['use_caching']

    Args:
        fun (Callable): the function to be cached

    Returns:
        Any: the cached or recomputed result of the method
    """
    @wraps(fun)
    def caching_wrapper(*args, **kwargs):
        filename = caching_path(args[0], fun)
        if get_current_context().options['use_caching']:
            # try to load result
            try:
                with open(filename, 'rb') as f:
                    return pickle.load(f)
            except FileNotFoundError:
                pass
            # recompute and save result
            result = fun(*args, **kwargs)
            file_folder = os.path.dirname(filename)
            if not os.path.isdir(file_folder):
                os.makedirs(file_folder)
            with open(filename, 'wb') as f:
                pickle.dump(result, f)
            return result
        else:
            # execute function unmodified
            return fun(*args, **kwargs)
    return caching_wrapper


def caching_path(exp: 'ManifoldMorExperiment', fun: Callable) -> str:
    return os.path.join(
        exp.get_experiment_folder(),
        PATH_CACHED_RESULTS,
        '{}.pkl'.format(fun.__name__),
    )


def exists_cached_result(exp: 'ManifoldMorExperiment', fun: Callable) -> bool:
    return os.path.exists(caching_path(exp, fun))
