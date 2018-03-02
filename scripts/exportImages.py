import argparse
import boto3
import botocore
import datetime
import os
import psycopg2
import re

from pathlib import Path

S3_KEY = os.environ['S3_ACCESS_KEY_ID']
ACCESS_KEY = os.environ['S3_SECRET_ACCESS_KEY']

def init():
    parser = argparse.ArgumentParser()
    default = 'new_batch' + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
    parser.add_argument('dir', nargs='?', default=default)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d','--dev', help='query dev db', action='store_true')
    group.add_argument('-t', '--test', help='query test db', action='store_true')
    group.add_argument('-p', '--production', help='query prod db', action='store_true')
    args = parser.parse_args()

    if args.dev:
        print('Dev config')
        db_url = os.environ['DEV_DATABASE_URL']
        bucket_name = os.environ['DEV_S3_BUCKET']
    if args.test:
        print('Test config')
        db_url = os.environ['TEST_DATABASE_URL']
        bucket_name = os.environ['TEST_S3_BUCKET']
    if args.production:
        print('Prod config')
        db_url = os.environ['PRD_DATABASE_URL']
        bucket_name = os.environ['PRD_S3_BUCKET']


    changeAndCreateDirs(args.dir)

    user, password, host, port, database = re.match('postgresql://(.*?):(.*?)@(.*?):(.*?)/(.*)', db_url).groups()
    getNonExportedImages(user, password, host, database, bucket_name)
    print("exiting")

def updateImageRefAfterExport(user, password, host, database, id):
    conn = psycopg2.connect( host=host, user=user, password=password, dbname=database )
    cur = conn.cursor()

    try:
        cur.execute("UPDATE image_refs SET exported = True WHERE id = %s", (id, ))
        conn.commit()
    except:
        raise
    cur.close()
    conn.close()

def getNonExportedImages(user, password, host, database, bucket_name):
    conn = psycopg2.connect( host=host, user=user, password=password, dbname=database )
    cur = conn.cursor()

    cur.execute("SELECT id, link, exported FROM image_refs WHERE exported = False")

    res = cur.fetchone()
    while res is not None:
        id, link, exported = res
        file_name = link.rsplit('/', 1)[-1]
        print(id, link, exported)

        try:
            pullResourceFromS3(bucket_name, file_name)
            print(file_name)
            updateImageRefAfterExport(user, password, host, database, id)
        except Exception as e:
            print(e)
            print(" at: " + file_name)
        res = cur.fetchone()

    cur.close()
    conn.close()

def pullResourceFromS3(bucket_name, file_name):

    s3 = boto3.resource(
            's3',
            aws_access_key_id=S3_KEY,
            aws_secret_access_key=ACCESS_KEY,
            )

    print("fetching: %s"  % file_name)

    try:
        s3.Object(bucket_name, file_name).download_file(file_name)
        #print("yupp, trying to dopwnload")
    except botocore.exceptions.ClientError as e:
        print(e.response)
        if e.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


def changeAndCreateDirs(dirname):
    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    parent_dir = Path(path).parent

    os.chdir(parent_dir)

    if not os.path.exists('data'):
        os.makedirs('data')
        print("made dir: %s" % 'data')

    os.chdir('data')

    print("pulling files into directory: %s " % dirname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    os.chdir(dirname)
    for i in range(10):
        classDir = 'c' + str(i)
        os.makedirs(classDir)
        print("created subdir: %s" % classDir)

if __name__ == "__main__":
    init()
