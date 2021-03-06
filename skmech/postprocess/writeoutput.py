"""writes output"""
import numpy as np
import os


def write_output(time, field, file_name,
                 exe_time,
                 datatype='Node'):
    """Write field output file

    Parameters
    ----------

    """
    # Create file if it does not exist or if it was created during
    # another execution
    # Cehck is file exists
    if os.path.isfile(f'{file_name}.out'):
        # True if file already exist
        if exe_time > os.path.getmtime(f'{file_name}.out'):
            # True if file was created before this execution
            # then overwrite previous file
            with open(f'{file_name}.out', 'w+') as out:
                out.write(f"{time}")
                if type(field) is np.ndarray:
                    for fld in field:
                        out.write(f" {fld}")
                    out.write("\r\n")
                else:
                    out.write(f" {field} \r\n")
        else:
            # file was created during this execution, append
            with open(f'{file_name}.out', 'a+') as out:
                out.write(f"{time}")
                if type(field) is np.ndarray:
                    for fld in field:
                        out.write(f" {fld}")
                    out.write("\r\n")
                else:
                    out.write(f" {field} \r\n")
    else:
        # file does not exist, create
        with open(f'{file_name}.out', 'w+') as out:
            out.write(f"{time}")
            if type(field) is np.ndarray:
                for fld in field:
                    out.write(f" {fld}")
                out.write("\r\n")
            else:
                out.write(f" {field} \r\n")
