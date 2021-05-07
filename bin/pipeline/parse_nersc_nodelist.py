#! /usr/bin/env python
"""
parse_nersc_nodelist.py
"""
from puzle.utils import parse_nersc_nodelist

if __name__ == '__main__':
    nodelist = parse_nersc_nodelist()
    nodelist_str = ' '.join(nodelist)
    print(nodelist_str)
