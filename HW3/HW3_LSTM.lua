-- Only requirements allowed
require("hdf5")
require("nn")
require("rnn")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'PTB_LSTM.hdf5', 'data file')
cmd:option('-classifier', 'lstm', 'classifier to use')

-- Hyperparameters
cmd:option('-din', 50, 'dimension of word embedding')
cmd:option('-dhid', 300, 'dimension of hidden layer')
cmd:option('-lambda', 0.01, 'learning rate')
cmd:option('-eps', 0.00001, 'epsilon at which lambda halves')
cmd:option('-epochs', 20, 'number of training epochs')
cmd:option('-seqlen', 50, 'sequential length for batches')
cmd:option('-dropout', 0.5, 'probability for dropout')

-- Utility functions
function adaptiveLambda(lambda, currLoss, prevLoss, eps)
  if prevLoss - currLoss < eps then
    print("Adapted!")
    return lambda/2
  else
    return lambda
  end
end

-- Models
function RNN(train_input, train_output, clf, nclasses, din, dhid, epochs, seqlen, lambda, eps, dropout)
  local nbatch, batch_size = train_input:size()[1], train_input:size()[2]
  local train_input_T = train_input:t() -- batch_size x nbatch
  local train_output_T = train_output:t()
  local valbatch = valid_input:size()[1]
  local valid_input_T = valid_input:t() -- batch_size x valbatch
  local valid_output_T = valid_output:t()
	local valid_output_table = {}
	for i = 1, batch_size do
		valid_output_table[i] = valid_output_T[i]
	end


  -- Initialize network (LSTM/RNN/GRU)
  local net = nn.Sequential()
  local LT = nn.LookupTable(nclasses, din) -- batch_size x nbatch x din
  net:add(LT)
  net:add(nn.SplitTable(1, 3)) -- batch_size table of nbatch x din
  if clf == 'lstm' then
    local seqLSTM = nn.Sequencer(nn.Sequential()
			:add(nn.LSTM(din, dhid)) -- batch_size table of nbatch x dhid
			:add(nn.Linear(dhid, nclasses)) -- batch_size table of nbatch x nclasses
			:add(nn.LogSoftMax()) -- batch_size table of nbatch x nclasses
		)
    seqLSTM:remember('both')
    net:add(seqLSTM)
	elseif clf == 'rnn' then
	  local rm = nn.Sequential() -- x is nbatch x din; s is nbatch x dhid
	  rm:add(nn.JoinTable(2)) -- [x,s] is nbatch x (din + dhid)
	  rm:add(nn.Linear(din + dhid, dhid)) -- [x,s]*W + b yields nbatch x dhid s'
	  rm:add(nn.Tanh()) -- tanh([x,s] * W + b)
	  local seqRNN = nn.Sequencer(nn.Sequential()
	    :add(nn.Recurrence(rm, dhid, 1))
	    :add(nn.Linear(dhid, nclasses))
			:add(nn.LogSoftMax())
	  )
	  seqRNN:remember('both')
	  net:add(seqRNN)
	else -- 'gru'
	  local seqGRU = nn.Sequencer(nn.Sequential()
			:add(nn.GRU(din, dhid))
			:add(nn.Linear(dhid, nclasses))
			:add(nn.LogSoftMax())
		)
	  seqGRU:remember('both')
	  net:add(seqGRU)
	end
  local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

  -- Get parameters
  params, gradParams = net:getParameters()
  for i = 1, params:size()[1] do
    params[i] = torch.uniform(-0.05, 0.05)
  end
  prevLoss = 100000

  -- Train network
  for t = 1, epochs do
		print(t)
		-- Run minibatches (seqlen)
		for j = 1, torch.ceil(batch_size / seqlen) do
			local train_input_mb = train_input_T[{{ (j-1) * seqlen + 1, math.min(j * seqlen, batch_size) }}]
			local train_output_mb = {}
			for i = 1, math.min(j * seqlen, batch_size) - (j-1) * seqlen do
				train_output_mb[i] = train_output_T[(j-1) * seqlen + i]
			end

			-- Manual SGD
      net:training()
			criterion:forward(net:forward(train_input_mb), train_output_mb)
			net:zeroGradParameters()
			net:backward(train_input_mb, criterion:backward(net.output, train_output_mb))
      gradParams = gradParams:map(torch.sign(gradParams), function(x, s) return s * math.min(math.abs(x), 5) end)
			net:updateParameters(lambda)
  	end
    -- Accuracy on development set
    net:evaluate()
    local val_pred = net:forward(valid_input_T)
    local currLoss = criterion:forward(val_pred, valid_output_table)
    print("RNN validation loss: " .. currLoss)
    --local val_perp = perplexity(val_pred, valid_output_T, 'rnn')
    --print("RNN validation perplexity: " .. val_perp)
    lambda = adaptiveLambda(lambda, currLoss, prevLoss, eps)
    print(lambda)
    prevLoss = currLoss
  end
  return net:forward(valid_input_T):t()
end


function main()
  -- Parse input params
  opt = cmd:parse(arg)
  local f = hdf5.open(opt.datafile, 'r')
	local nclasses = f:read('nclasses'):all():long()[1]
	local clf = opt.classifer

	-- Parse hyperparameters
  local din = opt.din
  local dhid = opt.dhid
  local epochs = opt.epochs
  local bsize = opt.bsize
  local seqlen = opt.seqlen
  local lambda = opt.lambda
  local eps = opt.eps
  local dropout = opt.dropout

  -- Load training data
  train_input = f:read('train_input'):all():long()
  train_output = f:read('train_output'):all():long()

  -- Load development data
  valid_input = f:read('valid_input'):all():long()
	valid_output = f:read('valid_output'):all():long()

  -- Train.
  predictions = RNN(train_input, train_output, clf, nclasses, din, dhid, epochs, seqlen, lambda, eps, dropout)

  -- Test.
	LTweights = net:get(1).weight
	local myFile = hdf5.open('LT_weights.hdf5', 'w')
	myFile:write('weights', LTweights)
	myFile:close()
end

main()
