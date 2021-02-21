# ---------------------------Tom Keane: 01788365------------------------------
import os
import shutil
import sqlite3
import csv
from datetime import datetime


def timer(func, *args, **kwargs):
    start = datetime.now()
    func(*args, **kwargs)
    time_d = (datetime.now() - start).total_seconds()
    return time_d


def collect_crime_data():
    if os.path.isdir('london_collected'):
        for filename in os.listdir('london_collected'):
            file_path = os.path.join('london_collected', filename)
            os.unlink(file_path)
    else:
        os.mkdir('london_collected')
    for root, dirs, files in os.walk('./london'):
        for name in files:
            full_path = os.path.join(root, name)
            shutil.copy2(full_path, 'london_collected')
    return


def create_london_db():
    filepaths = [os.path.join('london_collected', f) for f in os.listdir('london_collected')]
    conn = sqlite3.connect("london_db")
    c = conn.cursor()
    c.execute('drop table if exists crimes')
    c.execute("create table crimes(Crime_ID,Month,Reported_by,Falls_within,Longitude,"
              "Latitude,Location,LSOA_code,LSOA_name,Crime_type,Last_outcome_category,Context)")
    for file in filepaths:
        with open(file, "r") as f:
            readcsv = csv.reader(f)
            next(readcsv)
            c.executemany('INSERT INTO crimes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)', readcsv)
    conn.commit()
    conn.close()
    return


def run_crime_pipeline():
    start = datetime.now()
    no_files = len([name for name in os.listdir('./london')])
    print("Starting crime data extraction pipeline. \nCopying {} monthly csv files into london_collected directory...".format(no_files))
    time_d = timer(collect_crime_data)
    no_files_copied = len([name for name in os.listdir('./london_collected')])
    print("Copied {0} files in {1:.2f} seconds\nPopulating london_db database...".format(no_files_copied, time_d))
    time_d = timer(create_london_db)
    conn = sqlite3.connect("london_db")
    c = conn.cursor()
    c.execute("select count(*) from crimes")
    no_records = c.fetchone()[0]
    print("Completed Populating {0} records to crimes table in {1:.2f} seconds".format(no_records, time_d))
    print("Crime data extraction pipeline completed. Total running time: {0:.2f}".format((datetime.now() - start).total_seconds()))
    return
