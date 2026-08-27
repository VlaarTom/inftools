import glob
import importlib
import pathlib
from inspect import getmembers, getmodule, isfunction
from os.path import basename, isfile

MAXLINES = 100


def is_mod_function(mod, func):
    "checks that func is a public function defined in module mod"
    return (
        isfunction(func)
        and getmodule(func) == mod
        and not func.__name__.startswith("_")
    )


def list_functions(mod):
    "list of functions defined in module mod"
    return [
        func.__name__
        for func in mod.__dict__.values()
        if is_mod_function(mod, func)
    ]


def get_mapper(folders, mod_path):
    """List all functions given folders"""
    mapper = {}
    for folder in folders:
        fpath = mod_path + f"/{folder}"
        files = [basename(i)[:-3] for i in glob.glob(fpath + "/*py")]
        for file in files:
            mod = importlib.import_module(
                f".{folder}.{file}", package="inftools"
            )

            funcs = list_functions(mod)

            # get functions
            all_funcs = {i[0]: i[1] for i in getmembers(mod, isfunction)}
            for func in funcs:
                mapper[func] = all_funcs[func]  # func[1]
    return mapper


def dzlog(command, mod_path):
    """"""
    commands = [command]
    log_path = mod_path + "/misc/.log"
    if isfile(log_path):
        with open(log_path) as read:
            for line in read:
                commands.append(line)
    commands = list(set(commands))[-MAXLINES:]

    with open(log_path, "w") as write:
        for line in commands:
            write.write(line)


def log():
    """"""
    mod_path = str(pathlib.Path(__file__).parent.resolve())
    log_path = mod_path + "/.log"
    with open(log_path) as read:
        for line in read:
            print(line.rstrip())
