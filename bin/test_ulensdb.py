#! /usr/bin/env python
"""
test_ulensdb.py
"""

import numpy as np
import time
from puzle.ulensdb import insert_db_id, remove_db_id

time.sleep(abs(np.random.normal()))

insert_db_id()
time.sleep(10)
remove_db_id()
