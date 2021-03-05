#! /usr/bin/env python
"""
catalog.py
"""

import os
import psycopg2
from collections import namedtuple


def ulens_con():
    """Connection to the NERSC ulens database.

    Connection is opened to the database with:
        from puzle import catalog
        ulens_con = catalog.ulens_con()
    Connection is closed to the database with:
        ulens_con.close()
    The connection to the database should always be closed after completing
    the necessary transactions with the database.

    Args:
        None

    Returns:
        psycopg2.extensions.connection: Connection to the NERSC ulens database

    """

    if 'NERSC_HOST' in os.environ:
        return psycopg2.connect(dbname='ulens',
                                host='nerscdb03.nersc.gov',
                                port=5432,
                                user='ulens_admin',
                                password='frer334322222t3453')
    else:
        return psycopg2.connect(dbname='ulens',
                                host='localhost',
                                port=5432,
                                user='ulens_admin')


def query_ulens_db(query, con=None):
    """Sends a query to a PostgreSQL database, maintaining the state of the db.

    query_db should be used when the query DOES NOT alter the state of the
    database. This is a "read only" operation.

    Args:
        query : str
            Query to be sent to the PostgreSQL database. This query must be a
            single string and must therefore be wrapped in double quotes if the
            query requires single quotes, to specify a text value for instance.
        con : psycopg2.extensions.connection
            Connection object to the database using psycopg2. This is expected
            to be created through the ks_con function. For example:
                from puzle import catalog
                desi_con = catalog.desi_con()
                query_db(query, desi_con)
            If con == None, a connection object will be created and destroyed
            for this single execution.

    Returns:
        result : list of tuples
            The results of the query are returned as a list of tuples.
            There are three common use cases that require different wrapping
            around the query_db function to return usable results.
            Example 1: Returning a single parameter
                query = "SELECT count(*) FROM table"
                result = query_db(query, con)
                result = result[0][0] # count is a single value
            Example 2: Returning a list of single parameters
                query = "SELECT col1 FROM table;"
                result = query_db(query, con)
                result_list = [r[0] for in result] # result_list is a list
            Example 3: Returning a list of multiple parameters
                query = "SELECT col1,col2 FROM table;"
                result_list = query_db(query, con) # result_list is a list

    """
    # Obtain a connection if None is present
    close_con = False
    if con is None:
        con = ulens_con()
        close_con = True
    # Cursor allows code to execute commands and will be automatically closed
    # upon the exit from the query_db function.
    cursor = con.cursor()
    # Executes the query using the cursor.
    cursor.execute(query)
    # Fetches all rows of a query result.
    result = cursor.fetchall()
    # Close the connection if none was given to the function
    if close_con:
        con.close()
    return result


def query_ps1_psc(ra, dec, radius=2, con=None):
    radius_deg = radius / 3600.
    query = f'select obj_id, ra_stack, dec_stack, rf_score, quality_flag, ' \
            f'sqrt((ra_stack-({ra}))^2+(dec_stack-({dec}))^2) as dist from ps1_psc ' \
            f'where q3c_radial_query(ra_stack, dec_stack, {ra}, {dec}, {radius_deg}) ' \
            f'order by dist asc limit 1;'
    result = query_ulens_db(query, con=con)

    if len(result) == 1:
        result = result[0]
        rf_tuple = namedtuple('rf',
                              'obj_id ra_stack dec_stack rf_score quality_flag')
        rf = rf_tuple(obj_id=int(result[0]),
                      ra_stack=float(result[1]),
                      dec_stack=float(result[2]),
                      rf_score=float(result[3]),
                      quality_flag=int(result[4]))
    else:
        rf = None
    return rf
