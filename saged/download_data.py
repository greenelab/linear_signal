''' This script downloads the human compendium from refine.bio and the labels assigned to it
from the whistl repository (https://github.com/greenelab/whistl).
'''

import json
import os
import pathlib
import sys
import time
import zipfile

import requests
import wget

API_URL_BASE = 'https://api.refine.bio/v1/'


def unzip_results(result_path: pathlib.Path, data_dir: pathlib.Path) -> None:
    with zipfile.ZipFile(result_path, 'r') as zip_object:
        zip_object.extractall(data_dir)


def create_data_dir() -> pathlib.Path:
    """ Create the directory to store data in if it doesn't already exist

    Arguments
    ---------
    None

    Returns
    -------
    data_dir: The path to the data directory that was created
    """
    curr_file = pathlib.Path(__file__).absolute()
    repo_dir = curr_file.parents[1]
    data_dir = repo_dir.joinpath('data')

    try:
        os.mkdir(data_dir)
    except FileExistsError:
        pass

    return data_dir


def authenticate() -> str:
    """ Get an authentication token to allow the use of the refine.bio API. Doing so accepts the
    terms of use of refine.bio data

    Arguments
    ---------
    None

    Returns
    -------
    token: The authenticator to be used as a parameter in future calls to the API
    """
    response = requests.post(API_URL_BASE + 'token/')
    token_id = response.json()['id']
    response = requests.put('https://api.refine.bio/v1/token/' + token_id + '/',
                            json.dumps({'is_activated': True}),
                            headers={'Content-Type': 'application/json'})

    return token_id


def download_human_compendium(token: str,
                              compendium_id: int,
                              data_dir: pathlib.Path) -> pathlib.Path:
    """ Download the human compendium data from refine.bio and store it in the data directory

    Arguments
    ---------
    token: The authenticator to be used as a parameter in future calls to the API
    data_dir: The path to the data directory that was created
    compendium_id: The refine.bio compendium ID for the human compendium

    Returns
    -------
    compendium_path: The path to the newly downloaded compendium file
    """
    query_params = json.dumps({'primary_organism_name': 'HOMO_SAPIENS',
                               'compendium_version': 2,
                               'quant_sf_only': False
                             })
    headers = {'Content-Type': 'application/json',
               'API-KEY': token
              }
    query_url = API_URL_BASE + 'compendia/{}/'.format(compendium_id)

    response = requests.get(query_url, query_params, headers=headers).json()

    download_url = response['computed_file']['download_url']

    file_name = wget.download(download_url, out=str(data_dir))

    compendium_path = data_dir.joinpath(file_name)

    return compendium_path


def get_compendium_id(token: str) -> int:
    """ Find the correct refine.bio compendium to download human data from

    Arguments
    ---------
    token: The authenticator to be used as a parameter in future calls to the API

    Returns
    -------
    compendium_id: The refine.bio compendium ID for the human compendium
    """
    query_params = json.dumps({'primary_organism_name': 'HOMO_SAPIENS',
                               'compendium_version': 2,
                               'quant_sf_only': False
                             })
    headers = {'Content-Type': 'application/json',
               'API-KEY': token
              }

    done = False
    query_url = API_URL_BASE + 'compendia/'
    while not done:
        response = requests.get(query_url, query_params, headers=headers).json()
        results = response['results']
        for compendium_data in results:
            if (compendium_data['primary_organism_name'] == 'HOMO_SAPIENS' and
                not compendium_data['quant_sf_only']
               ):
                return compendium_data['id']

        query_url = response['next']

        if query_url is None:
            sys.stderr.write('Human compendium not found\n')
            sys.exit(1)

        # Don't hit the API too much
        time.sleep(.2)
    return None


if __name__ == '__main__':
    print('This script obtains a token from refine.bio as a method of accepting their '
          'terms of use. To avoid agreeing implicitly to the terms for you, we '
          'provide them here https://www.refine.bio/terms.')

    response = input('\nDo you agree to the terms of use? (Yes/No): ').lower()
    if response != 'yes' and response != 'y':
        print('Terms of use declined. Exiting...')
        sys.exit(0)

    refine_bio_token = authenticate()

    data_dir = create_data_dir()

    print('Downloading data. Warning, the compendium data file is around 50 GB zipped')

    compendium_id = get_compendium_id(refine_bio_token)

    out_file_path = download_human_compendium(refine_bio_token, compendium_id, data_dir)

    unzip_results(out_file_path, data_dir)
