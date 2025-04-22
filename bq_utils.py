# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 12:42:34 2023

@author: Carlo
"""

from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud.exceptions import NotFound
import json, os

def get_acct(filename):
    try:
        acct = json.loads(os.environ.get('BQ_SERVICE_ACCOUNT_KEY'))
        if acct is None:
            raise Exception

    except:
        with open(filename) as s:
            acct = json.load(s)

    return acct

def authenticate_bq(acct): #acct - service acct details, json type
    credentials = service_account.Credentials.from_service_account_info(
        acct
    )
    client = bigquery.Client(credentials=credentials,project=credentials.project_id)
    return client,credentials

def check_dataset(client,project_id,dataset_name): #check if dataset in BQ is already existing, create dataset if not
    datasets = [i.dataset_id for i in list(client.list_datasets())]
    try:
        if dataset_name not in datasets:
            platform_dataset = "{}.{}".format(project_id,dataset_name) #format "project_id.platform" ie "lica-rdbms.rapide"
            dataset = bigquery.Dataset(platform_dataset)
            dataset.location = "US"
            dataset = client.create_dataset(dataset, timeout=30)
            print("Created dataset {}".format(platform_dataset))
            datasets = [i.dataset_id for i in list(client.list_datasets())]
            print("Updated GCP-Bigquery datasets")
            print(datasets)
        else:
            print("{} already in GCP-Bigquery".format(dataset_name.title()))
        return True
    except:
        print ('Unable to create dataset.')
        return False

def check_table(client, table_id):
    status = False
    try:
        client.get_table(table_id)
        print ('Table {} already exists.'.format(table_id))
        status = True
    except NotFound:
        print ('Table {} is not found.'.format(table_id))
    finally:
        return status


def load_config(time_col, auto = True, write_disposition='WRITE_APPEND',
                src_format=bigquery.SourceFormat.CSV,
                allow_quoted_newlines =True,
                partition_type=bigquery.TimePartitioningType.DAY):

    if time_col is not None:
        return bigquery.LoadJobConfig(
                    #autodetect = auto,
                    write_disposition = write_disposition, #WRITE_TRUNCATE, WRITE_APPEND, WRITE_EMPTY
                    source_format = src_format,
                    allow_quoted_newlines = allow_quoted_newlines,
                    time_partitioning=bigquery.TimePartitioning(
                            type_= partition_type,
                            field= time_col,  # Name of the column to use for partitioning.
                            #expiration_ms=7776000000,  # 90 days.
                        )
                    )
    else:
        return bigquery.LoadJobConfig(
                    #autodetect = auto,
                    write_disposition = write_disposition, #WRITE_TRUNCATE, WRITE_APPEND, WRITE_EMPTY
                    source_format = src_format,
                    allow_quoted_newlines = allow_quoted_newlines
                    )

def bq_write(df,credentials,dataset_name,table_name,client, write_mode = 'WRITE_APPEND'):
    try:
        time_col = df.select_dtypes(include = 'datetime').columns[-1]
    except:
        time_col = None

    try:
        job_config = load_config(time_col, auto = True,
                                 write_disposition = write_mode,
                                 src_format = bigquery.SourceFormat.CSV)
    except:
        job_config = load_config(time_col = None, auto = True,
                                 write_disposition = write_mode,
                                 src_format = bigquery.SourceFormat.CSV)

    target_table_id = "{}.{}.{}".format(credentials.project_id,dataset_name,table_name)#project_id.dataset_id.table_id - "lica-rdbms.rapide.MarketBasket"

    try:
        job = client.load_table_from_dataframe(df, target_table_id, job_config=job_config)# upload table
    except:
        # no datetime
        job = client.load_table_from_dataframe(df, target_table_id)# upload table

    # while job.state != "DONE":
    #     time.sleep(2)
    #     job.reload()

    # print(job.result())
    table = client.get_table(target_table_id)
    report = 'Loaded {} rows and {} columns to {}'.format(
        table.num_rows, len(table.schema), target_table_id)

    return report

def write_bq(client, credentials, table_id, data,
             autodetect : bool = True,
             write_mode : str = 'WRITE_APPEND'):
    # check if table exists
    if check_table(client, table_id):
        pass
    # create table if does not exist
    else:
        table = client.create_table(bigquery.Table(table_id))
        print (f'Table {table_id} created.')

    try:
        # set loadjobconfig
        job_config = bigquery.LoadJobConfig()
        # auto_detect schema
        job_config.autodetect = autodetect

        # source format
        job_config.source_format = bigquery.SourceFormat.CSV
        # allow quoted new lines
        job_config.allow_quoted_newlines = True
        # write disposition
        if write_mode in ['WRITE_APPEND', 'WRITE_EMPTY']:
            job_config.write_disposition = write_mode.upper()
            # allow field addition
            job_config.schema_update_options = [
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
            ]

        elif write_mode == 'WRITE_TRUNCATE':
            job_config.write_disposition = write_mode.upper()

        else:
            job_config.write_disposition = 'WRITE_APPEND'
            job_config.schema_update_options = [
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
            ]

    except Exception as e:
        raise e

    # load data to table
    job = client.load_table_from_dataframe(data,
                                           table_id,
                                           job_config=job_config)# upload table

    job.result()
    print("Loaded {} rows into {}.".format(job.output_rows, table_id))
    return job

def query_bq(messages_table_id,client):
    messages_columns = ['thread_id','created_at','from_name','msg_body','to_name'] #columns to query
    messages_query = """
                        SELECT * FROM `{}`
                    """.format(messages_table_id)#standard SQL query
    return client.query(messages_query).to_dataframe() #returns dataframe


def init_bq(table_name : str = 'gulong_customer_chat') -> dict:
    '''
    Loads BigQuery project info

    Returns
    -------
        - bq_dict : dict
            keys contain client, credentials, and project id

    '''

    # load account object from json file
    acct = get_acct('absolute-gantry-363408-1f5b5b4dc774.json')
    # get client and creds via authentication of acct
    client, credentials = authenticate_bq(acct)
    # check table
    project_id = 'absolute-gantry-363408'
    dataset = 'carmax_webscrape'
    table_name = 'fb_marketplace' if table_name is None else table_name
    table_id = '.'.join([project_id, dataset, table_name])

    return {'client' : client,
            'credentials' : credentials,
            'table_id' : table_id}

def load_save_data(bq_dict : dict,
                   df : None,
                   ls : str = 'load',
                   mode : str = 'WRITE_APPEND'):
    '''
    Load/Save pandas dataframe to csv

    Args:
    -----
        - bq_dict : dict
            keys containing client, credentials, project and table id
        - df : pd.DataFrame, default is None
            dataframe to save
        - ls : str, default
        - mode : str
            'a' for append, 'w' for truncation

    '''

    if ls == 'load':
        try:
            output = query_bq(bq_dict['table_id'],
                              bq_dict['client'])

        except Exception as e:
            print(e)
            output = None

        finally:
            return output

    elif ls == 'save':
        job = write_bq(client = bq_dict['client'],
                        credentials = bq_dict['credentials'],
                        table_id = bq_dict['table_id'],
                        data = df,
                        write_mode = mode
                        )
        return None

    else:
        return None

def sql_query_bq(query, acct):
    client, credentials = authenticate_bq(acct)

    query_job = client.query(query)
    df = query_job.to_dataframe()

    return df