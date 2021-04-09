# ! /usr/bin/env python
"""
generate_sources_maps.py
"""
import glob
import pickle

from puzle.utils import return_DR5_dir


def generate_sources_maps():
    DR5_dir = return_DR5_dir()
    folders = glob.glob(f'{DR5_dir}/sources_*')
    source_fnames = []
    for folder in folders:
        source_fnames += glob.glob(folder + '/sources.*.txt')

    for i, source_fname in enumerate(source_fnames):
        print(source_fname, i, len(source_fnames))
        f = open(source_fname, 'r')
        sources_map = {}
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            file_position = f.tell() - len(line)
            source_id = line.split(',')[0]
            sources_map[source_id] = file_position
        f.close()

        map_filename = source_fname.replace('.txt', '.sources_map')
        with open(map_filename, 'wb') as fileObj:
            pickle.dump(sources_map, fileObj)


if __name__ == '__main__':
    generate_sources_maps()
