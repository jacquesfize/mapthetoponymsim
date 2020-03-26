import os
import time
import re

import numpy as np
import pandas as pd


def read_geonames(file):
    """
    Return a dataframe that contains Geonames data.
    
    Parameters
    ----------
    file : str
        path of the Geonames Csv file
    
    Returns
    -------
    pd.DataFrame
        geonames data
    """
    dtypes_dict = {
        0: int,  # geonameid
        1: str,  # name
        2: str,  # asciiname
        3: str,  # alternatenames
        4: float,  # latitude
        5: float,  # longitude
        6: str,  # feature class
        7: str,  # feature code
        8: str,  # country code
        9: str,  # cc2
        10: str,  # admin1 code
        11: str,  # admin2 code
        12: str,  # admin3 code
        13: str,  # admin4 code
        14: int,  # population
        15: str,  # elevation
        16: int,  # dem (digital elevation model)
        17: str,  # timezone
        18: str,  # modification date yyyy-MM-dd
    }
    rename_cols = {
        0: "geonameid",  # geonameid
        1: "name",  # name
        2: "asciiname",  # asciiname
        3: "alternatenames",  # alternatenames
        4: "latitude",  # latitude
        5: "longitude",  # longitude
        6: "feature_class",  # feature class
        7: "feature_code",  # feature code
        8: "country_code",  # country code
        9: "cc2",  # cc2
        10: "admin1_code",  # admin1 code
        11: "admin2_code",  # admin2 code
        12: "admin3_code",  # admin3 code
        13: "admin4_code",  # admin4 code
        14: "population",  # population
        15: "elevation",  # elevation
        16: "dem",  # dem (digital elevation model)
        17: "timezone",  # timezone
        18: "modification_date",  # modification date yyyy-MM-dd
    }
    data = pd.read_csv(
        file,
        sep="\t",
        header=None,
        quoting=3,
        dtype=dtypes_dict,
        na_values="",
        keep_default_na=False,
        error_bad_lines=False,
    )
    data.rename(columns=rename_cols, inplace=True)
    return data


def parse_title_wiki(title_wiki):
    """
    Parse Wikipedia title
    
    Parameters
    ----------
    title_wiki : str
        wikipedia title
    
    Returns
    -------
    str
        parsed wikipedia title
    """
    return re.sub("\(.*\)", "", str(title_wiki)).strip().lower()


def _split(lst, n, complete_chunk_value):
    """
    Split a list into chunk of n-size.
    
    Parameters
    ----------
    lst : list
        input list
    n : int
        chunk size
    complete_chunk_value : object
        if last chunk size not equal to n, this value is used to complete it
    
    Returns
    -------
    list
        chunked list
    """
    chunks = [lst[i : i + n] for i in range(0, len(lst), n)]
    if not chunks:
        return chunks
    if len(chunks[-1]) != n:
        chunks[-1].extend([complete_chunk_value] * (n - len(chunks[-1])))
    return np.array(chunks)


class Chronometer:
    def __init__(self):
        self.__task_begin_timestamp = {}

    def start(self, task_name):
        """
        Start a new task chronometer
        
        Parameters
        ----------
        task_name : str
            task id
        
        Raises
        ------
        ValueError
            if a running task already exists with that name
        """
        if task_name in self.__task_begin_timestamp:
            raise ValueError(
                "A running task exists with the name {0}!".format(task_name)
            )
        self.__task_begin_timestamp[task_name] = time.time()

    def stop(self, task_name):
        """
        Stop and return the duration of the task
        
        Parameters
        ----------
        task_name : str
            task id
        
        Returns
        -------
        float
            duration of the task in seconds
        
        Raises
        ------
        ValueError
            if no task exist with the id `task_name`
        """
        if not task_name in self.__task_begin_timestamp:
            raise ValueError("The {0} task does not exist!".format(task_name))
        
        duration = time.time() - self.__task_begin_timestamp[task_name]
        del self.__task_begin_timestamp[task_name]

        return duration


from keras.callbacks import Callback
import time

class EpochTimer(Callback):
    def __init__(self,log_filename):
        self.epoch = 0
        self.timer = time.time()
        self.output = open(log_filename,'w')
        self.output.write("{0},{1}\n".format("Epoch","Execution Time"))
        self.output.flush()

    def on_epoch_begin(self,epoch, logs={}):
        self.timer = time.time()

    def on_epoch_end(self, epoch, logs=None):
        end_time = time.time() - self.timer
        self.output.write("{0},{1}\n".format(self.epoch,end_time))
        self.output.flush()
        self.epoch += 1 

if __name__ == "__main__":
    chrono = Chronometer()
    chrono.start("test")
    chrono.start("test2")
    time.sleep(3)
    print(chrono.stop("test"))
    time.sleep(3)
    print(chrono.stop("test2"))
