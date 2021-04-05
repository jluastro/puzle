#! /usr/bin/env python
"""
test_ulensdb.py
"""

import time
from puzle.ulensdb import insert_db_id, remove_db_id
from puzle.utils import get_logger

logger = get_logger(__name__)

insert_db_id()
time.sleep(10)
remove_db_id()
