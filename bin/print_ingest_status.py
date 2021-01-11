import glob
from puzle import db
from puzle.models import SourceIngestJob, StarIngestJob


def print_source_exporting_status():
    jobs = db.session.query(SourceIngestJob).filter(SourceIngestJob.finished == True).all()
    job_ids_db = [job.id for job in jobs]
    job_ids_db.sort()

    folders = glob.glob('sources_*')
    fis = []
    for folder in folders:
        fis += glob.glob(folder + '/sources.*.txt')
    job_ids_disk = [int(f.split('.')[-2]) for f in fis]

    job_ids_remaining = list(set(job_ids_db) - set(job_ids_disk))

    print('\nExporting sources...')
    print('%i jobs - db' % len(job_ids_db))
    print('%i jobs - disk' % len(job_ids_disk))
    print('%i jobs - remaining' % len(job_ids_remaining))


def print_source_uploading_status():
    jobs = db.session.query(SourceIngestJob).filter(SourceIngestJob.finished == True).all()
    jobs = db.session.query(SourceIngestJob).filter(SourceIngestJob.finished == True).all()
    job_ids = [job.id for job in jobs]
    job_ids_uploaded = [job.id for job in jobs if job.uploaded == True]

    job_ids_remaining = list(set(job_ids) - set(job_ids_uploaded))

    folders = glob.glob('sources_*')
    fis = []
    for folder in folders:
        fis += glob.glob(folder + '/sources.*.txt')
    job_ids_disk = [int(f.split('.')[-2]) for f in fis]

    job_ids_waiting_to_upload = list(set(job_ids_disk).intersection(set(job_ids_remaining)))

    print('\nUploading sources...')
    print('%04d jobs - finished on db' % len(job_ids))
    print('%04d jobs - uploaded on db' % len(job_ids_uploaded))
    print('%04d jobs - remaining to upload' % len(job_ids_remaining))
    print('%04d jobs --- on disk to upload' % len(job_ids_waiting_to_upload))


def print_source_processing_status():
    all_jobs = db.session.query(SourceIngestJob).all()
    jobs = db.session.query(SourceIngestJob).filter(SourceIngestJob.started == True).all()
    job_ids = [job.id for job in jobs]
    job_ids_finished = [job.id for job in jobs if job.finished == True]

    job_ids_processing = list(set(job_ids) - set(job_ids_finished))

    folders = glob.glob('sources_*')
    fis = []
    for folder in folders:
        fis += glob.glob(folder + '/sources.*.txt')
    job_ids_disk = [int(f.split('.')[-2]) for f in fis]
    job_ids_disk = list(set(job_ids_disk).intersection(set(job_ids)))

    job_ids_disk_processing = list(set(job_ids_disk) - set(job_ids_finished))

    print('\nProcessing sources...')
    print('%04d jobs - total on db' % len(all_jobs))
    print('%04d jobs - started on db' % len(job_ids))
    print('%04d jobs - finished on db' % len(job_ids_finished))
    print('%04d jobs - processing' % len(job_ids_processing))
    print('%04d jobs --- on disk total' % len(job_ids_disk))
    print('%04d jobs --- on disk processing' % len(job_ids_disk_processing))


def print_star_uploading_status():
    jobs = db.session.query(StarIngestJob).filter(StarIngestJob.finished == True).all()
    job_ids = [job.source_ingest_job_id for job in jobs]
    job_ids_uploaded = [job.source_ingest_job_id for job in jobs if job.uploaded == True]

    job_ids_remaining = list(set(job_ids) - set(job_ids_uploaded))

    folders = glob.glob('stars_*')
    fis = []
    for folder in folders:
        fis += glob.glob(folder + '/stars.*.txt')
    job_ids_disk = [int(f.split('.')[-2]) for f in fis]

    job_ids_waiting_to_upload = list(set(job_ids_disk).intersection(set(job_ids_remaining)))

    print('\nUploading stars...')
    print('%04d jobs - finished on db' % len(job_ids))
    print('%04d jobs - uploaded on db' % len(job_ids_uploaded))
    print('%04d jobs - remaining to upload' % len(job_ids_remaining))
    print('%04d jobs --- on disk to upload' % len(job_ids_waiting_to_upload))


def print_star_processing_status():
    all_jobs = db.session.query(StarIngestJob).all()
    jobs = db.session.query(StarIngestJob).filter(StarIngestJob.started == True).all()
    job_ids = [job.source_ingest_job_id for job in jobs]
    job_ids_finished = [job.source_ingest_job_id for job in jobs if job.finished == True]

    job_ids_processing = list(set(job_ids) - set(job_ids_finished))

    folders = glob.glob('stars_*')
    fis = []
    for folder in folders:
        fis += glob.glob(folder + '/stars.*.txt')
    job_ids_disk = [int(f.split('.')[-2]) for f in fis]
    job_ids_disk = list(set(job_ids_disk).intersection(set(job_ids)))

    job_ids_disk_processing = list(set(job_ids_disk) - set(job_ids_finished))

    print('\nProcessing stars...')
    print('%04d jobs - total on db' % len(all_jobs))
    print('%04d jobs - started on db' % len(job_ids))
    print('%04d jobs - finished on db' % len(job_ids_finished))
    print('%04d jobs - processing' % len(job_ids_processing))
    print('%04d jobs --- on disk total' % len(job_ids_disk))
    print('%04d jobs --- on disk processing' % len(job_ids_disk_processing))


def print_all_status():
    # print_source_exporting_status()
    print_source_processing_status()
    print_source_uploading_status()
    print_star_processing_status()
    print_star_uploading_status()


if __name__ == '__main__':
    print_all_status()
