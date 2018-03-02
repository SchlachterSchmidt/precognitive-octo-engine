"""Refeed manually captured data to the classifier."""
import argparse
import datetime
import os
import re

from pathlib import Path

import boto3
import botocore
import psycopg2

S3_KEY = os.environ['S3_ACCESS_KEY_ID']
ACCESS_KEY = os.environ['S3_SECRET_ACCESS_KEY']

def init():
    """Initial method call to set variables and start query."""
    parser = argparse.ArgumentParser()
    default = 'new_batch' + '_' + datetime.datetime.now().strftime("%Y%m%d%H%M")
    parser.add_argument('dir', nargs='?', default=default)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-d', '--dev', help='query dev db', action='store_true')
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


    change_and_create_dir(args.dir)

    user, password, host, port, database = re.match(
        'postgresql://(.*?):(.*?)@(.*?):(.*?)/(.*)', db_url).groups()
    get_non_exported_images(user, password, host, database, bucket_name)
    print("exiting")

def update_image_ref_after_export(user, password, host, database, image_id):
    """Update image ref to mark as exported."""
    conn = psycopg2.connect(host=host, user=user, password=password, dbname=database)
    cur = conn.cursor()

    try:
        cur.execute("UPDATE image_refs SET exported = True WHERE id = %s", (image_id, ))
        conn.commit()
    except:
        raise
    cur.close()
    conn.close()

def get_non_exported_images(user, password, host, database, bucket_name):
    """Query databse for non-exported images."""
    conn = psycopg2.connect(host=host, user=user, password=password, dbname=database)
    cur = conn.cursor()

    cur.execute("SELECT id, link, exported FROM image_refs WHERE exported = False")

    res = cur.fetchone()
    while res is not None:
        image_id, link, exported = res
        file_name = link.rsplit('/', 1)[-1]
        print(image_id, link, exported)

        try:
            pull_from_s3(bucket_name, file_name)
            print(file_name)
            update_image_ref_after_export(
                user, password, host, database, image_id)
        except Exception as exception:
            print(exception)
            print(" at: " + file_name)
        res = cur.fetchone()

    cur.close()
    conn.close()

def pull_from_s3(bucket_name, file_name):
    """Pull image resource from S3."""
    s3 = boto3.resource(
        's3',
        aws_access_key_id=S3_KEY,
        aws_secret_access_key=ACCESS_KEY,
        )

    print("fetching: %s"  % file_name)

    try:
        s3.Object(bucket_name, file_name).download_file(file_name)
        #print("yupp, trying to dopwnload")
    except botocore.exceptions.ClientError as exception:
        print(exception.response)
        if exception.response['Error']['Code'] == "404":
            print("The object does not exist.")
        else:
            raise


def change_and_create_dir(dirname):
    """Util method to safely change and create nonexisting dirs."""
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
        class_dir = 'c' + str(i)
        os.makedirs(class_dir)
        print("created subdir: %s" % class_dir)

if __name__ == "__main__":
    init()
