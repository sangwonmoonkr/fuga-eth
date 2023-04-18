const FugaController = artifacts.require("FugaController");
// _maxRound, _self_centered, _FitNum, _EvalNum, _batch_size, _learning_rate, _local_epochs, _val_steps
const maxRound = 5;
const self_centered = true;
const FitNum = 2;
const EvalNum = 2;
const batch_size = 1;
const learning_rate = "0.00005"
const local_epochs = 1;
const val_steps = 5;



module.exports = function(deployer) {
  deployer.deploy(FugaController, maxRound, self_centered, FitNum, EvalNum, batch_size, learning_rate, local_epochs, val_steps);
};

