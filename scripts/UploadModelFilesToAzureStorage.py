import os, uuid
import argparse
from pathlib import Path
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

try:
    # Parse program parameters
    parser = argparse.ArgumentParser(description='Program parameters needed on this application')

    parser.add_argument('--model_name', required=True, type=str, help='The unique name of the model')
    parser.add_argument('--xml_file_path', required=True, type=str, help='The path of xml file')
    parser.add_argument('--bin_file_path', required=True, type=str, help='The path of bin file')
    parser.add_argument('--connection_string', required=False, type=str, help='The connection string to access Azure BLOB storage', default="AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;DefaultEndpointsProtocol=http;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;")

    args = parser.parse_args()

    model_name = args.model_name

    # Retrieve the connection string for use with the application.
    connect_str = args.connection_string

    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # Create a unique name for the container
    container_name = "ovms"

    # Create a file in the local data directory to upload and download
    upload_xml_file_path = args.xml_file_path
    upload_xml_file_name = model_name + "/1/" + Path(upload_xml_file_path).name
    upload_bin_file_path = args.bin_file_path
    upload_bin_file_name = model_name + "/1/" + Path(upload_bin_file_path).name

    # Create a blob client using the local xml file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=upload_xml_file_name)

    print("\nUploading to Azure Storage as blob:\n\t" + upload_xml_file_name)

    # Upload the created file
    with open(upload_xml_file_path, "rb") as data:
        blob_client.upload_blob(data)

    # Create a blob client using the local bin file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=upload_bin_file_name)

    print("\nUploading to Azure Storage as blob:\n\t" + upload_bin_file_name)

    # Upload the created file
    with open(upload_bin_file_path, "rb") as data:
        blob_client.upload_blob(data)

except Exception as ex:
    print('Exception:')
    print(ex)

