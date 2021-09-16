def multi_run_wrapper(args):
    return do(*args)


def get_hello():
    return "hello"


def get_bye():
    return "bye"


def get_lol():
    return "lol"


def do(task):
    re = ""
    if task == "h":
        re = get_hello()
    elif task == "b":
        re = get_bye()
    elif task == "l":
        re = get_lol()
    return re


if __name__ == "__main__":
    from multiprocessing import Pool
    pool = Pool(4)
    results = pool.map(multi_run_wrapper,[("l"),("b"),("h")])
    print results