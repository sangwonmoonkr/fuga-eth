// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./PeerSystem.sol";

contract FugaController is PeerSystem {
    event ServerMessage(string field, address sender);
    event getConfigMessage(bool self_centered, uint batch_size, string learning_rate, uint local_epochs, uint val_steps);
    event getClientMessage(string model_hash, string[] dweight_hash, uint num_sample, uint score);
    // event FitInsMessage(string[] model_hashes, string[] dweight_hashes, uint[] num_samples, uint[] scores);
    event FitInsMessage(string[] dweight_hashes, uint[] num_samples, uint[] scores);
    event EvaluateInsMessage(string[] model_hashes);

    enum Status{
        Ready,
        Config,
        FitIns,
        FitRes,
        EvaluateIns,
        EvaluateRes
    }

    struct Config {
        bool self_centered;
        uint maxRound;
        uint FitNum;
        uint EvalNum;
        uint batch_size;
        string learning_rate;
        uint local_epochs;
        uint val_steps;
    }
    Config private config;

    struct Client {
        Status status;
        string model_hash;
        string[] dweight_hash;
        uint num_sample;
        uint score;
        uint updateCount;
        uint[] evals;
    }

    constructor(uint _maxRound, bool _self_centered, uint _FitNum, uint _EvalNum, uint _batch_size, string memory _learning_rate, uint _local_epochs, uint _val_steps){
        stage = Stage.Ready;
        currentRound = 1;
        setConfig(_maxRound, _self_centered, _FitNum, _EvalNum, _batch_size, _learning_rate, _local_epochs, _val_steps);
    }

    mapping(address => Client) private client;

    modifier checkStatus(Status _status){
        require(_status==client[msg.sender].status,"Status not match");
        _;
    }

    function getClient() external {
    // view returns(string memory, uint, uint){
        emit getClientMessage(client[msg.sender].model_hash, client[msg.sender].dweight_hash, client[msg.sender].num_sample, client[msg.sender].score);
        // return (client[msg.sender].model_hash, client[msg.sender].num_sample, client[msg.sender].score);
    }

    function getConfig() onlyCurrentClients currentStage(Stage.Config) checkStatus(Status.Ready) external {
    // view returns(bool, uint, string memory, uint, uint){
        emit getConfigMessage(config.self_centered, config.batch_size, config.learning_rate, config.local_epochs, config.val_steps);
        // return (config.self_centered, config.batch_size, config.learning_rate, config.local_epochs, config.val_steps);
    }

    function setConfig(uint _maxRound, bool _self_centered, uint _FitNum, uint _EvalNum, uint _batch_size, string memory _learning_rate, uint _local_epochs, uint _val_steps) internal {
        config.self_centered = _self_centered;
        config.maxRound = _maxRound;
        config.FitNum = _FitNum;
        config.EvalNum = _EvalNum;
        config.batch_size = _batch_size;
        config.learning_rate = _learning_rate;
        config.local_epochs = _local_epochs;
        config.val_steps = _val_steps;
    }

    function joinRound() currentStage(Stage.Ready) external {
        require(clientRound[msg.sender] < currentRound, "Client already joined this round.");
        client[msg.sender].status = Status.Ready;
        clientRound[msg.sender] = currentRound;
        currentClients.push(msg.sender);
        lastJoinTime = block.timestamp;
        // emit ServerMessage("JoinRound", msg.sender);
    }

    function ConfigRes() external onlyCurrentClients currentStage(Stage.Config) checkStatus(Status.Ready) {
        client[msg.sender].status = Status.Config;
        currentCompelete++;
        if(currentCompelete == currentClients.length){
            stage = Stage.Fit;
            currentCompelete = 0;
            distributePeers(config.FitNum);
            emit ServerMessage("FitIns", msg.sender);
        }
    }

    function FitIns() external onlyCurrentClients currentStage(Stage.Fit) checkStatus(Status.Config) {
        address[] memory peers = clientPeers[msg.sender];
        uint numPeers = peers.length;
        // string[] memory model_hashes = new string[](numPeers);
        string[] memory dweight_hashes = new string[](numPeers);
        uint[] memory num_samples = new uint[](numPeers);
        uint[] memory scores = new uint[](numPeers);
        for(uint i=0; i<numPeers; i++) {
            // model_hashes[i] = client[peers[i]].model_hash;
            if(client[peers[i]].updateCount > 0)
                dweight_hashes[i] = client[peers[i]].dweight_hash[client[peers[i]].updateCount - 1];
            num_samples[i] = client[peers[i]].num_sample;
            scores[i] = client[peers[i]].score;
        }
        client[msg.sender].status = Status.FitIns;
        // return (model_hashes, num_samples, scores);
        // emit FitInsMessage(model_hashes, dweight_hashes, num_samples, scores);
        emit FitInsMessage(dweight_hashes, num_samples, scores);
    }

    function FitRes(string memory _dweight_hash, string memory _model_hash, uint _num_sample) external onlyCurrentClients currentStage(Stage.Fit) checkStatus(Status.FitIns) {
        client[msg.sender].model_hash = _model_hash;
        client[msg.sender].dweight_hash.push(_dweight_hash);
        client[msg.sender].num_sample = _num_sample;
        client[msg.sender].updateCount++;
        client[msg.sender].status = Status.FitRes;
        currentCompelete++;
        if(currentCompelete == currentClients.length){
            stage = Stage.Evaluate;
            currentCompelete = 0;
            distributePeers(config.EvalNum);
            emit ServerMessage("EvaluateIns", msg.sender);
        }
    }

    function EvaluateIns() external onlyCurrentClients currentStage(Stage.Evaluate) checkStatus(Status.FitRes) {
        uint numPeers = clientPeers[msg.sender].length;
        string[] memory model_hashes = new string[](numPeers);
        for(uint i=0; i<numPeers; i++) {
            model_hashes[i] = client[clientPeers[msg.sender][i]].model_hash;
        }
        delete client[msg.sender].evals;
        client[msg.sender].status = Status.EvaluateIns;
        // return (model_hashes);
        emit EvaluateInsMessage(model_hashes);
    }

    function EvaluateRes(string[] memory _model_hashes, uint[] memory _values) external onlyCurrentClients currentStage(Stage.Evaluate) checkStatus(Status.EvaluateIns) {
        uint numPeers = _model_hashes.length;
        for(uint i=0; i<numPeers; i++) {
            for(uint j=0; j<numPeers; j++) {
                Client storage peer = client[clientPeers[msg.sender][(i+j)%numPeers]];
                if(keccak256(abi.encodePacked(peer.model_hash)) == keccak256(abi.encodePacked(_model_hashes[i]))) {
                    peer.evals.push(_values[i]);
                    continue;
                }
            }
        }
        currentCompelete++;
        client[msg.sender].status = Status.EvaluateRes;
        if(currentCompelete == currentClients.length){
            stage = Stage.Finished;
            currentCompelete = 0;
            finishRound();
        }
    } 


    function sort(uint[] memory data) internal pure returns (uint[] memory) {
        // Bubble sort the data in ascending order
        uint n = data.length;
        for (uint i = 0; i < n; i++) {
            for (uint j = i+1; j < n; j++) {
                if (data[i] > data[j]) {
                    uint temp = data[i];
                    data[i] = data[j];
                    data[j] = temp;
                }
            }
        }
        return data;
    }

    function medianEval() private {
        for(uint i=0; i<currentClients.length; i++) {
            uint[] memory evals = client[currentClients[i]].evals;
            uint n = evals.length;
            uint[] memory sortedData = sort(evals);
            uint median = n % 2 == 0 ? (sortedData[n/2 - 1] + sortedData[n/2]) / 2 : sortedData[n/2];
            client[currentClients[i]].score = median;
        }
    }

    function normScore() private {
        uint totalScore = 0;
        for(uint i=0; i<currentClients.length; i++) {
            totalScore += client[currentClients[i]].score;
        }
        for(uint i=0; i<currentClients.length; i++) {
            client[currentClients[i]].score = client[currentClients[i]].score * 100 / totalScore;
        }
    }

    function finishRound() currentStage(Stage.Finished) private {
        medianEval();
        normScore();
        delete currentClients;
        currentRound++;
        if(currentRound > config.maxRound) {
            stage = Stage.Finished;
            emit ServerMessage("Finished", msg.sender);
        }
        else{
            stage = Stage.Ready;
            emit ServerMessage("Ready", msg.sender);
        }
    }

    function startRound() external onlyCurrentClients {
        if(lastJoinTime + 20 - 5 <= block.timestamp && stage==Stage.Ready) {
            stage = Stage.Config;
            emit ServerMessage("ConfigIns", msg.sender);
        }
    }

    function Test() external {
        emit ServerMessage("Ready", msg.sender);
    }
} 