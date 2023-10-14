import functools
import pickle
import os
import hashlib
import inspect
import shutil

def pickle_lru_cache(maxsize=None, filename=None, purge_cache = True):
    def decorator(func):
        nonlocal filename
        func_hash = hashlib.md5(inspect.getsource(func).encode()).hexdigest()

        if filename is None:
            filename = func.__name__+".cache.pickle"
            filename_backup = filename+".bak"
        
        if not os.path.exists(filename):
            with open(filename, 'wb') as f:
                pickle.dump({"__hash__": func_hash}, f)

        cache = {}
            
        with open(filename, 'rb') as f:
            cache = pickle.load(f)
        
        if func_hash != cache["__hash__"]:
            if purge_cache:
                print("Source code has changed, cache is purged")
                cache.clear()
                with open(filename, 'wb') as f:
                    pickle.dump({"__hash__": func_hash}, f)
            else:
                print("Source code has changed, but cache is not purged!")
                with open(filename, 'wb') as f:
                    cache["__hash__"] = func_hash
                    pickle.dump(cache, f)


        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            if key in cache:
                print(f"{func.__name__} result is read from cache file.")
                return cache[key]
            
            result = func(*args, **kwargs)
            
            if maxsize is not None and len(cache) >= maxsize:
                oldest_key = next(iter(cache))
                del cache[oldest_key]

            cache[key] = result
            with open(filename, 'wb') as f:
                shutil.copyfile(filename, filename_backup)
                pickle.dump(cache, f)
            
            return result
        
        return wrapper

    return decorator

if __name__ == "__main__":
    @pickle_lru_cache(maxsize=5)
    def expensive_function(x ,y):
        print(f"Calculating for {x}")
        return x ** 2
    
    print(expensive_function(2, 1))
    print(expensive_function(3, 2))
    print(expensive_function(4, 1))
    print(expensive_function(2, 2))
    print(expensive_function(2, 1))
