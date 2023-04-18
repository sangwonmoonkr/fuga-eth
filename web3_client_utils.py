
import hashlib
import io
import os
import boto3
from dotenv import load_dotenv
import torch
import numpy as np

# MNEMONIC = os.environ.get('MNEMONIC')
BUCKET_NAME = 'fugaeth'

# def get_private_key(MNEMONIC):
#     w3 = web3.Web3()
#     w3.eth.account.enable_unaudited_hdwallet_features()
#     account = w3.eth.account.from_mnemonic(MNEMONIC, account_path="m/44'/60'/0'/0/0")
#     return account.key

# PRIVATE_KEY = get_private_key(MNEMONIC)

def send_transaction(w3, contract, function, PUBLIC_KEY, PRIVATE_KEY):
    """
    Sends a transaction to the blockchain.
    """

    # create the transaction
    tx = {
        'chainId': w3.eth.chainId,
        'from': PUBLIC_KEY,
        'to': contract.address,
        'gas': 2000000,
        'gasPrice': w3.eth.gasPrice,
        'nonce' :w3.eth.getTransactionCount(PUBLIC_KEY),
        'data': function._encode_transaction_data()
    }

    # sign the transaction
    signed_tx = w3.eth.account.sign_transaction(tx, private_key=PRIVATE_KEY)

    # send the transaction
    tx_hash = w3.eth.sendRawTransaction(signed_tx.rawTransaction)

    # wait for the transaction to be mined
    tx_recipt = w3.eth.waitForTransactionReceipt(tx_hash)

    return tx_recipt

def read_transaction(w3, contract, function_name, PUBLIC_KEY, PRIVATE_KEY):
    function = getattr(contract.functions, function_name)()
    tx_receipt = send_transaction(w3, contract, function, PUBLIC_KEY, PRIVATE_KEY)
    if function_name=='getConfig':
        event_logs = contract.events.getConfigMessage().processReceipt(tx_receipt)
    elif function_name=='getClient':
        event_logs = contract.events.getClientMessage().processReceipt(tx_receipt)
    elif function_name=='FitIns':
        event_logs = contract.events.FitInsMessage().processReceipt(tx_receipt)
    elif function_name=='EvaluateIns':
        event_logs = contract.events.EvaluateInsMessage().processReceipt(tx_receipt)
    return event_logs[0]['args']

def make_hash(params):
    # Concatenate all arrays in the list of parameters
    concatenated_array = np.concatenate([param.flatten() for param in params])

    # Convert the concatenated array to bytes and calculate the hash
    hash_bytes = hashlib.sha256(concatenated_array.tobytes()).digest()

    # Convert the hash to a string and return it
    return hash_bytes.hex()

def read_dweight(dweight_hash):
    with open(f'./dweights/{dweight_hash}.bin', 'rb') as f:
        buffer = io.BytesIO(f.read())
        param = torch.load(buffer)
    return param

def save_dweight(dweight_hash, dweight):
    buffer = io.BytesIO()
    torch.save(dweight, buffer)
    serialized_dweight = buffer.getvalue()
    with open(f'./dweights/{dweight_hash}.bin', 'wb') as f:
        f.write(serialized_dweight)

def check_dweight(dweight_hash):
    dweight = read_dweight(dweight_hash)
    if not make_hash(dweight)==dweight_hash:
        print("hash value is not matched")
        return False
    else:
        return True

def read_model(model_hash):
    with open(f'./models/{model_hash}.bin', 'rb') as f:
        buffer = io.BytesIO(f.read())
        param = torch.load(buffer)
    return param

def save_model(model_hash, parameters_prime):
    buffer = io.BytesIO()
    torch.save(parameters_prime, buffer)
    serialized_model = buffer.getvalue()
    with open(f'./models/{model_hash}.bin', 'wb') as f:
        f.write(serialized_model)

def check_model(model_hash):
    param = read_model(model_hash)
    if not make_hash(param)==model_hash:
        print("hash value is not matched")
        return False
    else:
        return True

def fit_dweight(client, old_params):
    new_params, num_examples_train, results = client.fit(old_params)
    # Calculate the weight changes
    weight_changes = [new - old for new, old in zip(new_params, old_params)]
    return weight_changes, new_params, num_examples_train, results


def aggregate_fit(client, dweight_hashes, num_samples, scores):
    dweight = []
    print("original scores : ",scores)
    for dweight_hash in dweight_hashes:
        dweight.append(read_dweight(dweight_hash)) 
        
    # Normalize the evaluation scores
    sum_scores = sum(scores)
    norm_scores = [score / sum_scores for score in scores]
    print("normalized scores : ",norm_scores)

    # Combine normalized evaluation scores with the dataset portion
    sum_samples = sum(num_samples)
    norm_samples = [sample / sum_samples for sample in num_samples]
    print("normalized samples : ",norm_samples)

    # Calculate the combined weights
    weights = [np.linalg.norm([score, sample]) for score, sample in zip(norm_scores, norm_samples)]

    # Normalize the combined weights
    norm_weights = [weight / sum(weights) for weight in weights]
    print("normalized weights : ",norm_weights)

    # Calculate the weighted model updates
    weighted_updates = [w * update for w, update in zip(norm_weights, dweight)]
    new_updates = [sum(updates) for updates in zip(*weighted_updates)]

    # Update the model parameters
    new_params = [param + update for param, update in zip(client.get_parameters(), new_updates)]

    # return client.fit(new_params)   
    return fit_dweight(client, new_params)
