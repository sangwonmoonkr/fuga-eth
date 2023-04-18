import argparse
import json
from typing import Callable, Dict, Tuple
import web3
from imdbClient import IMDBClient

from handle import *

def s3_connection():
    try:
        s3 = boto3.client(
            service_name="s3",
            region_name="ap-northeast-2",
            aws_access_key_id=os.environ.get('S3_ACCESS_KEY'),
            aws_secret_access_key=os.environ.get('S3_SCRETE_KEY')
        )
    except Exception as e:
        print(e)
    else:
        print("s3 bucket connected!")
        return s3


def listen_for_event(contract, event_name):
    # create a filter to listen for the specified event
    event_filter = contract.events[event_name].createFilter(fromBlock='latest')

    while True:
        # check if any new events have been emitted
        for event in event_filter.get_new_entries():
            try:
                # if the specified event has been emitted, return its message
                if event.event == event_name:
                    yield event.args
            except Exception as e:
                print(f"Error processing event: {e}")
        # wait for new events
        time.sleep(5)

def web3_connection(s3, w3, contract, test) -> Tuple[Callable[[], Dict], Callable[[Dict], None]]:
    """
    Creates a connection to the blockchain and returns a function to receive messages and a function to send messages.
    """
    web3_message_iterator = listen_for_event(contract, "ServerMessage")

    def receive():
        try:
            return next(web3_message_iterator)
        except StopIteration:
            return None

    def send(msg):
        return handle_send(msg, s3, w3, contract, PUBLIC_KEY, PRIVATE_KEY, test)

    return (receive, send)


def start_web3_client(client, contract_address, abi, network, test):
    if network == 'mumbai':
        HTTP_PROVIDER = os.environ.get('MUMBAI_HTTP_PROVIDER')
    else:
        HTTP_PROVIDER = 'http://localhost:7545'

    while True:
        if not test:
            s3 = s3_connection()
        else:
            s3 = None
        w3 = web3.Web3(web3.HTTPProvider(HTTP_PROVIDER))
        contract = w3.eth.contract(address=contract_address, abi=abi)
        receive,send = web3_connection(s3, w3, contract, test)
        # receive,send = next(conn)

        print("client ready!")

        while True:
            server_message = receive()
            print("server message : ",server_message)
            if(server_message is not None):
                client_message, sleep_duration, keep_going = handle_receive(
                    client, server_message, s3, w3, contract, PUBLIC_KEY, PRIVATE_KEY, test
                )
                if(client_message is not None):
                    send(client_message)
        
                if not keep_going:
                    break

        # Check if we should disconnect and shut down
        if sleep_duration == 0:
            print("Disconnect and shut down")
            break
        # Sleep and reconnect afterwards
        print("Sleeping for {} seconds".format(sleep_duration))
        time.sleep(sleep_duration)

if __name__ == "__main__":
    # get the contract address and abi from the config file'
    parser = argparse.ArgumentParser()
    # parser.add_argument('-c','--contract_address', required=True, type=str, default='config.json')
    parser.add_argument('-pb','--public_key', type=str, default= os.environ.get('PUBLIC_KEY'))
    parser.add_argument('-pr','--private_key', type=str, default= os.environ.get('PRIVATE_KEY'))
    parser.add_argument('-n','--network', type=str, default='ganache')
    parser.add_argument('-t','--test', type=bool, default=False)
    args = parser.parse_args()

    # contract_address = args.contract_address
    contract_address = input("contract address : ")
    abi = json.load(open('contract/build/contracts/FugaController.json', 'r'))['abi']

    PUBLIC_KEY = args.public_key
    PRIVATE_KEY = args.private_key
    print(PUBLIC_KEY, PRIVATE_KEY)

    # get the client
    client = IMDBClient()

    # start the client
    start_web3_client(client, contract_address, abi, args.network, args.test)