from web3_client_utils import *
import time

def handle_receive(client, msg, s3, w3, contract, PUBLIC_KEY, PRIVATE_KEY, test):
    """
    Handles a received message.
    Returns a tuple containing the response message, the number of samples and a boolean indicating whether the client should shut down.
    """
    # check the field of the message
    field = msg['field']

    if field == "Finished":
        print("finished")
        return None, 0, False

    # if the message is a request for Ready, join the round
    if field == "Ready":
        print("join round")
        function = getattr(contract.functions, 'joinRound')()
        send_transaction(w3, contract, function, PUBLIC_KEY, PRIVATE_KEY)
        time.sleep(1*20)
        print("call start round")
        function = getattr(contract.functions, 'startRound')()
        send_transaction(w3, contract, function, PUBLIC_KEY, PRIVATE_KEY)
        return None, 0, True

    # if the message is a request for the ConfigIns, set the Config data
    if field == "ConfigIns":
        print("config start")
        # receive the Config data
        response = read_transaction(w3, contract, 'getConfig', PUBLIC_KEY, PRIVATE_KEY)
        
        self_centered = response['self_centered']
        batch_size = response['batch_size']
        learning_rate = float(response['learning_rate'])
        local_epochs = response['local_epochs']
        val_steps = response['val_steps']

        config = {'self_centered': self_centered, 'batch_size': batch_size, 'learning_rate' : learning_rate ,'local_epochs': local_epochs, 'val_steps': val_steps}

        client.set_config(config)
        print("config set : ",config)

        return {'field':'ConfigRes'}, 0, True
        
    # if the message is a request for the FitIns, aggregate the model and return the result after fit
    if field == "FitIns":
        print("fit start")
        # receive the Client data
        response = read_transaction(w3, contract, 'getClient', PUBLIC_KEY, PRIVATE_KEY)

        # self_model_hash = response['model_hash']
        self_dweight_hash = response['dweight_hash']
        self_num_sample = response['num_sample']
        self_score = response['score']

        # model_hashes = []
        dweight_hashes = []
        num_samples = []
        scores = []

        response = read_transaction(w3, contract, 'FitIns', PUBLIC_KEY, PRIVATE_KEY)

        # other_model_hashes = response['model_hashes']
        other_dweight_hashes = response['dweight_hashes']
        other_num_samples =response['num_samples']
        other_scores = response['scores']


        # if self_model_hash is empty, fit the model with the initial parameters
        if(len(self_dweight_hash)==0):
            print("self dweight is empty")
            # fitres = client.fit(client.get_parameters())
            fitres = fit_dweight(client,client.get_parameters())
        # if self_model_hash is not empty, aggregate the model
        else:
            for i, dweight_hash in enumerate(other_dweight_hashes):
                if dweight_hash == '':
                    continue
                object_key = f'dweights/{dweight_hash}.bin'
                file_path = object_key
                if not test:
                    s3.download_file(BUCKET_NAME, object_key, file_path)

                # check hash value
                if(check_dweight(dweight_hash)):
                    # model_hashes.append(model_hash)
                    dweight_hashes.append(dweight_hash)
                    num_samples.append(other_num_samples[i])
                    scores.append(other_scores[i])

            if(check_dweight(self_dweight_hash[-1])):
                # model_hashes.append(self_model_hash)
                dweight_hashes.append(self_dweight_hash[-1])
                num_samples.append(self_num_sample)
                scores.append(sum(scores) if client.config['self_centered'] else self_score)

            # aggregate the model
            # fitres = aggregate_fit(client, model_hashes, num_samples, scores)
            fitres = aggregate_fit(client, dweight_hashes, num_samples, scores)

        # return the result of fit
        return {'field':'FitRes', 'data': fitres}, 0, True
    
    # if the message is a request for the EvaluateIns, evaluate the model and return the result
    elif field == "EvaluateIns":
        print("eval start")
        # evaluate the model on client
        response = read_transaction(w3, contract, 'EvaluateIns', PUBLIC_KEY, PRIVATE_KEY)
        model_hashes = response['model_hashes']
        evalres = {}

        for model_hash in model_hashes:
            object_key = f'models/{model_hash}.bin'
            file_path = object_key

            if not test:
                s3.download_file(BUCKET_NAME, object_key, file_path)

            # check hash value
            check_model(model_hash)
            param = read_model(model_hash)

            # evaluate the model
            loss, _, _ = client.evaluate(param)
            evalres[model_hash] = 1/loss
        
        print("evaluation result : ",evalres)

        return {'field':'EvaluateRes', 'data':evalres}, 0 , True

    else:
        print("unknown message")
        return None, 0, True

def handle_send(msg, s3, w3, contract, PUBLIC_KEY, PRIVATE_KEY, test):
    """
    Handles a message to be sent.
    Returns the transaction receipt.
    """
    # check the field of the message
    field = msg['field']

    # If the message is a ConfigRes, upload the result to the blockchain
    if field == 'ConfigRes':
        print("config complete")
        function = getattr(contract.functions, 'ConfigRes')()

    # If the message is a FitRes, save model and upload model hash to the blockchain
    if field == 'FitRes':
        print("fit complete")
        # get the message params
        dweights_prime, parameters_prime, num_examples_train, results = msg['data']

        # hash the message params which is dictionary
        model_hash = make_hash(parameters_prime)
        dweight_hash = make_hash(dweights_prime)

        save_model(model_hash, parameters_prime)
        save_dweight(dweight_hash, dweights_prime)

        # save and upload the model
        if not test:
            # specify the bucket name and object key
            object_key = f'models/{model_hash}.bin'

            # upload the file to S3
            with open(f'./models/{model_hash}.bin', 'rb') as f:
                s3.upload_fileobj(f, BUCKET_NAME, object_key)

            object_key = f'dweights/{dweight_hash}.bin'

            with open(f'./dweights/{dweight_hash}.bin', 'rb') as f:
                s3.upload_fileobj(f, BUCKET_NAME, object_key)

        # prepare the arguments for the FitRes function
        args = [dweight_hash, model_hash, num_examples_train]

        # get the function object from the contract ABI
        function = getattr(contract.functions, 'FitRes')(*args)

    # If the message is a EvaluateRes, upload the result to the blockchain
    elif field == 'EvaluateRes':
        print("eval complete")
        # get the message params
        evalres = msg['data']
        model_hashes = list(evalres.keys())
        values = [int(value) for value in evalres.values()]
        args = [model_hashes, values]

        # get the function object from the contract ABI
        function = getattr(contract.functions, 'EvaluateRes')(*args)

    send_transaction(w3, contract, function, PUBLIC_KEY, PRIVATE_KEY)

