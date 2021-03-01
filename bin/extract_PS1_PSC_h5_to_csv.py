import h5py
import glob
import os


def _return_csv_filename(fi):
    return fi.replace('.h5', '.csv')


def init_csv(fi):
    fi_csv = _return_csv_filename(fi)
    if os.path.exists(fi_csv):
        os.remove(fi_csv)

    with open(fi_csv, 'w') as f:
        f.write('obj_id,ra_stack,dec_stack,rf_score,quality_flag\n')


def export_rows(fi, objIDs, raStacks, decStacks, rfScores, qualityFlags):
    fi_csv = _return_csv_filename(fi)
    num_entries = len(objIDs)
    with open(fi_csv, 'a') as f:
        for i in range(num_entries):
            objID = objIDs[i][0]
            raStack = raStacks[i]
            decStack = decStacks[i]
            rfScore = rfScores[i]
            qualityFlag = qualityFlags[i][0]
            f.write('%s,%s,%s,%s,%s\n' % (objID,
                                          raStack,
                                          decStack,
                                          rfScore,
                                          qualityFlag))


def extract_files():
    fis = glob.glob('dec*classifications.h5')
    fis.sort()

    num_chunk = int(1e6)

    for i, fi in enumerate(fis):
        print('Extracting %s (%i / %i)' % (fi, i+1, len(fis)))
        with h5py.File(fi, 'r') as f:
            num_rows = f['class_table']['block0_values'].shape[0]
            init_csv(fi)
            for row_low in range(0, num_rows, num_chunk):
                row_high = row_low + num_chunk
                objIDs = f['class_table']['block1_values'][row_low:row_high]
                raStacks = f['class_table']['block0_values'][row_low:row_high, 0]
                decStacks = f['class_table']['block0_values'][row_low:row_high, 1]
                rfScores = f['class_table']['block0_values'][row_low:row_high, 2]
                qualityFlags = f['class_table']['block2_values'][row_low:row_high]
                export_rows(fi, objIDs, raStacks, decStacks, rfScores, qualityFlags)


if __name__ == '__main__':
    extract_files()
