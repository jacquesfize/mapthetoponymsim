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
