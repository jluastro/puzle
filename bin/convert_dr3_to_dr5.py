#! /usr/bin/env python
"""
convert_dr3_to_dr4.py
"""

"""
ZTF DR4 extends the public observations to 1/29/2021,
including another the last summer of data that could include microlensing
events in the galactic plane. However sources have already been
identified using DR3. This script converts those results into DR5.

It is only the sources that need to be converted, as they contain
but the object ID and the lightcurve position of each of their objects.
These object IDs are (thankfully) consistent between data releases,
however the addition of different data points shifts the lightcurve
positions. This conversion can be completed using the object maps
which have object IDs as keys and their lightcurve positions as sources.
"""

import os
import glob
import shutil
from zort.lightcurveFile import LightcurveFile

from puzle.utils import lightcurve_file_to_field_id

ulens_ztf_dir = '/global/cfs/cdirs/uLens/ZTF'


def fetch_unconverted_files_dr3(file_type):
    if file_type not in ['star', 'source']:
        print('file_type must be either star or source')
        return
    # grab source files in the DR3 folder
    files = []
    folders = glob.glob(f'{ulens_ztf_dir}/DR3/{file_type}s*')
    folders.sort()
    for folder in folders:
        fis = glob.glob(f'{folder}/{file_type}*')
        fis.sort()
        for fi in fis:
            folder_dr5 = folder.replace('DR3', 'DR5')
            fi_base = os.path.basename(fi)
            # if the source file already exists in DR5, skip
            if not os.path.exists(f'{folder_dr5}/{fi_base}'):
                files.append(fi)
    return files


def construct_lightcurve_filenames_dr5_dict():
    # grab DR4 lightcurve filenames sorted by field_ID
    # these can be slightly different than DR3 due to ra, dec bounds
    lightcurve_filenames_dr5_dict = {}
    for fi in glob.glob(f'{ulens_ztf_dir}/DR5/field*txt'):
        field = lightcurve_file_to_field_id(fi)
        lightcurve_filenames_dr5_dict[field] = fi
    return lightcurve_filenames_dr5_dict


def convert_sources_dr3_to_dr5():
    lightcurve_filenames_dr5_dict = construct_lightcurve_filenames_dr5_dict()

    lightcurveFile_dct = {}
    source_files_dr3 = fetch_unconverted_files_dr3(file_type='source')
    for i, source_file_dr3 in enumerate(source_files_dr3):
        if len(lightcurveFile_dct) >= 100:
            lightcurveFile_dct = {}
        print('Converting %s (%i/%i)' % (source_file_dr3, i, len(source_files_dr3)))
        lines_dr3 = open(source_file_dr3, 'r').readlines()
        header = lines_dr3[0]

        num_sources = len(lines_dr3) - 1
        num_missing = 0

        lines_dr5 = []
        skipFlag = False
        for line in lines_dr3[1:]:
            # get the dr5 lightcurve filename
            lightcurve_filename_dr3 = line.split(',')[7]
            field_id = lightcurve_file_to_field_id(lightcurve_filename_dr3)
            lightcurve_filename_dr5 = lightcurve_filenames_dr5_dict[field_id]

            # either load the lightcurveFile and cache it, or load from cache
            try:
                if lightcurve_filename_dr5 not in lightcurveFile_dct:
                    lightcurveFile = LightcurveFile(lightcurve_filename_dr5)
                    lightcurveFile_dct[lightcurve_filename_dr5] = lightcurveFile
                else:
                    lightcurveFile = lightcurveFile_dct[lightcurve_filename_dr5]
            except FileNotFoundError:
                print('---- Skipping %s due to missing objects_map file' % source_file_dr3)
                skipFlag = True
                break

            line_split = line.split(',')

            # looping over g, r and i
            for object_id_idx in [1, 2, 3]:
                object_id_dr3 = line_split[object_id_idx]
                lightcurve_position_idx = object_id_idx + 3

                # if the object ID doesn't exist, nothing to replace
                if object_id_dr3 == 'None':
                    continue

                # there could be objects that are no longer in the field due to the shifted boundaries
                # in this case, simply zero out the object with a None
                try:
                    lightcurve_position_dr5 = str(lightcurveFile.objects_map[int(object_id_dr3)])
                except KeyError:
                    source_id = line_split[0]
                    print('---- source %s missing from DR4 lightcurve file' % source_id)
                    line_split[object_id_idx] = 'None'
                    line_split[lightcurve_position_idx] = 'None'
                    num_missing += 1
                    continue

                # replace the DR3 lightcurve position with DR4
                line_split[lightcurve_position_idx] = lightcurve_position_dr5

            # join together new converted line
            new_line = ','.join(line_split)

            # replace the DR3 lightcurve filename with DR4
            new_line = new_line.replace(lightcurve_filename_dr3,
                                        lightcurve_filename_dr5)

            # keep the line for DR4 source file
            lines_dr5.append(new_line)

        if skipFlag:
            continue

        # create the DR4 source folder (if DNE)
        source_file_dr5 = source_file_dr3.replace('DR3', 'DR5')
        source_folder = os.path.dirname(source_file_dr5)
        if not os.path.exists(source_folder):
            os.makedirs(source_folder)

        # write the DR4 sources out to disk
        with open(source_file_dr5, 'w') as f:
            f.write(header)
            for line in lines_dr5:
                f.write(line)

        print(f'-- conversion complete: {num_missing} of {num_sources} sources missing')


def copy_stars_dr3_to_dr5():
    star_files_dr3 = fetch_unconverted_files_dr3(file_type='star')
    for i, star_file_dr3 in enumerate(star_files_dr3):
        print('Copying %s (%i/%i)' % (star_file_dr3, i, len(star_files_dr3)))
        star_file_dr5 = star_file_dr3.replace('DR3', 'DR5')
        star_folder_dr5 = os.path.dirname(star_file_dr5)
        if not os.path.exists(star_folder_dr5):
            os.makedirs(star_folder_dr5)
        shutil.copy(star_file_dr3, star_file_dr5)


if __name__ == '__main__':
    convert_sources_dr3_to_dr5()
    copy_stars_dr3_to_dr5()
