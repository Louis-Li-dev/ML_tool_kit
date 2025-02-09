import numpy as np
from typing import Union, Any
def process_numpy_str(
        data: Any, 
        to_numpy: bool = True
    ) -> Union[np.ndarray, list]:
    '''
        Parse the string into a numpy array
        Args:
            data (Any): data being parsed into a numpy array or list.
            to_numpy (bool), default to True: whether the return value is a numpy array.

        Returns:
            np.ndarray (np.ndarray, list): the prossed array(s)  
    '''
    if isinstance(data, list) or isinstance(data, np.ndarray):
        result = list(list(map(int, row.strip('[]').split())) for row in data)
    else:   
        result = list(map(int, data.strip("[]").split())) 
    return np.array(result) if to_numpy else result
    