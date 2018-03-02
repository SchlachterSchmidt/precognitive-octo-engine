"""Refeed manually captured data to the classifier."""
import argparse
from base64 import b64encode
import json
import os
from pathlib import Path
import requests


def init():

    parser = argparse.ArgumentParser()
    host_group = parser.add_mutually_exclusive_group(required=True)
    host_group.add_argument('-l',
                            '--localhost',
                            help='use localhost',
                            action='store_const',
                            const='localhost',
                            default='localhost',
                            dest='host')
    host_group.add_argument('-i',
                            '--ip',
                            help='destination IP address',
                            dest='host')
    parser.add_argument('-p',
                        '--port',
                        help='port number',
                        default=7000,
                        dest='port')
    parser.add_argument('-u',
                        '--user',
                        help='username and password',
                        required=True,
                        nargs=2,
                        metavar=('USER', 'PASS'))
    parser.add_argument('-b',
                        '--batch',
                        help='batch name',
                        required=True,
                        nargs=1,
                        dest='batch')
    args = parser.parse_args()

    port = args.port
    host = args.host
    uri = 'api/v0.1/classifier'
    url = "http://" + str(host) + ":" + str(port) + "/" + uri

    full_path = os.path.realpath(__file__)
    path, filename = os.path.split(full_path)
    parent_dir = Path(path).parent

    data_dir = parent_dir.joinpath('data')

    if not os.path.exists(data_dir):
        print("data directory does not exist\n"
              "aborting..")
        quit()
    os.chdir(data_dir)

    batch = args.batch[0]
    batch_dir = data_dir.joinpath(batch)
    if not os.path.exists(batch_dir):
        print("batch directory does not exist\n"
              "aborting..")
        quit()
    os.chdir(batch_dir)

    user_as_bytes = str.encode(args.user[0] +':' + args.user[1])
    b64_user_and_credentials = str(b64encode(user_as_bytes))[2:-1]

    prev_score = 1
    count = len(os.listdir(batch_dir))
    print("found %d images to refeed" % count)

    for image_file in os.listdir(batch_dir):
        if image_file == '.DS_Store':
            continue
        print(image_file)

        with open(image_file, 'rb') as image:
            headers = dict(Authorization="Basic " +
                           b64_user_and_credentials,
                           Content_type="multipart/form-data")
            # image data as byte stream, in 'data' field of request
            multipart_form_data = {'data': (image_file.lower(),
                                            open(image_file, 'rb'),
                                            'multipart/form-data')}
            values = {'prev_score': prev_score}


        response = requests.post(url,
                                 files=multipart_form_data,
                                 data=values,
                                 headers=headers)
        response_body = json.loads(response.text)
        prev_score = response_body['score']
        prediction = response_body['prediction']
        probs = response_body['probabilities']

        print(prev_score)
        print(prediction)
        print(probs)
        count = count - 1
        print("remaining: %d" % count)


if __name__ == "__main__":
    init()
