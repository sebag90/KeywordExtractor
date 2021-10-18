import os
from pathlib import Path


def progress_bar(
        iteration, total, prefix='', suffix='', decimals=1, length=40,
        fill='#', miss=".", end="\r", stay=True, fixed_len=False):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        miss        - Optional  : bar missing charachter (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    if fixed_len:
        bar_len = length - len(prefix) - len(suffix)
    else:
        bar_len = length

    percent = f"{100*(iteration/float(total)):.{decimals}f}"
    filled_length = int(bar_len * iteration // total)
    bar = f"{fill * filled_length}{miss * (bar_len - filled_length)}"
    to_print = f"\r{prefix} [{bar}] {percent}% {suffix}"
    print(to_print, end=end)

    # Print New Line on Complete
    if iteration >= total:
        if stay:
            print()
        else:
            # clean line given lenght of lase print
            print(" "*len(to_print), end=end)


def retrieve_files(pathstring):
    """
    given a path, this function collects all
    files and returns a list of paths

    """
    if pathstring[-1] == "/":
        pathstring = pathstring[:-1]
    file_list = []

    for path, _, files in os.walk(pathstring):
        for file in files:
            if path == pathstring:
                filepath = Path(f"{path}/{file}")
                file_list.append(filepath)

    return file_list
