from puzle import db
from puzle.models import StarProcessJob


def print_star_processing_status():
    all_jobs = db.session.query(StarProcessJob).all()
    jobs = db.session.query(StarProcessJob).filter(StarProcessJob.started == True).all()
    job_ids = [job.source_ingest_job_id for job in jobs]
    job_ids_finished = [job.source_ingest_job_id for job in jobs if job.finished == True]

    print('\nProcessing stars...')
    print('%04d jobs - total on db' % len(all_jobs))
    print('%04d jobs - started on db' % len(job_ids))
    print('%04d jobs - finished on db (%.2f %%)' % (len(job_ids_finished),
                                                    100 * float(len(job_ids_finished)) / len(all_jobs)))


if __name__ == '__main__':
    print_star_processing_status()
