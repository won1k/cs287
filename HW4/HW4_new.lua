-- Only requirement allowed
require("hdf5")
require("rnn")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'PTB.hdf5', 'data file')
cmd:option('-classifier', 'nnlm', 'classifier to use')

-- Hyperparameters
cmd:option('-n', 3, 'n-gram parameter')
cmd:option('-smooth', 0, 'smoothing parameter (alpha if clf = Laplace, D if clf = kn')
cmd:option('-din', 15, 'dimension of word embedding')
cmd:option('-dhid', 100, 'dimension of hidden layer')
cmd:option('-lambda', 0.01, 'learning rate')
cmd:option('-eps', 0.00001, 'epsilon at which lambda halves')
cmd:option('-epochs', 20, 'number of training epochs')
cmd:option('-bsize', 32, 'batch size')
cmd:option('-seqlen', 50, 'sequential length for batches')
cmd:option('-dropout', 0.5, 'probability for dropout')

-- Utility functions
function perplexity(predictions, output, clf)
  if clf == 'rnn' then
    local batch_size, nbatch = output:size()[1], output:size()[2]
    local N = nbatch * batch_size
    local logPP = 0
    for b = 1, nbatch do
      for i = 1, batch_size do
        if output[i][b] == 0 then
          logPP = logPP - torch.log(1 - predictions[i][b][1]) / N
        else
          logPP = logPP - torch.log(predictions[i][b][1]) / N
        end
      end
    end
    return torch.exp(logPP)
  else
    local N = output:size()[1]
    local logPP = 0
    for i = 1, N do
      if output[i] == 0 then
        logPP = logPP - torch.log(1 - predictions[i][1]) / N
      else
        logPP = logPP - torch.log(predictions[i][1]) / N
      end
    end
    return torch.exp(logPP)
  end
end

function adaptiveLambda(lambda, currLoss, prevLoss, eps)
  if prevLoss - currLoss < eps then
    print("Adapted!")
    return lambda/2
  else
    return lambda
  end
end

-- Models
function CountBased(train_input, train_output, n, smooth)
  local nrows, batch_size  = train_input:size()[1], train_input:size()[2]
  -- Construct "n-gram" windows
  local train_input_all = train_input:reshape(nrows * batch_size)
  local train_output_all = train_output:reshape(nrows * batch_size)
  local nsamples = nrows * batch_size - n
  local input_windows = torch.LongTensor(nsamples, n)
  local output_windows = torch.LongTensor(nsamples)
  for i = 1, nsamples do
    input_windows[i] = train_input_all[{{ i, i + n - 1 }}]
    output_windows[i] = train_output_all[i + n - 1]
  end

  -- Construct count matrix
  local dims = torch.LongStorage(n+1):fill(nfeatures)
  dims[n+1] = 2
  local F = torch.Tensor(dims):fill(smooth)

  -- Count occurrences
  for i = 1, nsamples do
    F[torch.totable(input_windows[i])][output_windows[i]] = F[torch.totable(input_windows[i])][output_windows[i]] + 1
  end

  -- Normalize
  local net = nn.Sum(n+1)
  dims[n+1] = 1
  local row_sums = net:forward(F):resize(dims)
  F = torch.cdiv(F, row_sums:expandAs(F))

  -- Predictions / Validation accuracy
  local valrows, batch_size  = valid_input:size()[1], valid_input:size()[2]
  local valid_input_all = valid_input:reshape(valrows * batch_size)
  local valid_output_all = valid_output:reshape(valrows * batch_size)
  local valsamples = valrows * batch_size - n
  local valid_input_windows = torch.LongTensor(valsamples, n)
  local valid_output_windows = torch.LongTensor(valsamples)
  for i = 1, valsamples do
    valid_input_windows[i] = valid_input_all[{{ i, i + n - 1 }}]
    valid_output_windows[i] = valid_output_all[i + n - 1]
  end
  local pred_probs = torch.Tensor(valsamples, 2)
  for i = 1, valsamples do
    pred_probs[i] = F[torch.totable(valid_input_windows[i])]
  end
  local values, predictions = torch.max(pred_probs, 2)
  local accuracy = 0
  local base_acc = 0
  for i = 1, valsamples do
    if predictions[i][1] == valid_output_windows[i] then
      accuracy = accuracy + 1
    end
    if valid_output_windows[i] == 1 then
      base_acc = base_acc + 1
    end
  end
  print(accuracy / valsamples)
  print(base_acc / valsamples)
  return predictions
end

function NNLM(train_input, train_output, n, din, dhid, epochs, bsize, lambda, eps)
  local nrows, batch_size  = train_input:size()[1], train_input:size()[2]
  -- Construct "n-gram" windows
  local train_input_all = train_input:reshape(nrows * batch_size)
  local train_output_all = train_output:reshape(nrows * batch_size)
  local nsamples = nrows * batch_size - n
  local input_windows = torch.LongTensor(nsamples, n)
  local output_windows = torch.DoubleTensor(nsamples)
  for i = 1, nsamples do
    input_windows[i] = train_input_all[{{ i, i + n - 1 }}]
    output_windows[i] = train_output_all[i + n - 1]
  end

  local valrows, batch_size  = valid_input:size()[1], valid_input:size()[2]
  local valid_input_all = valid_input:reshape(valrows * batch_size)
  local valid_output_all = valid_output:reshape(valrows * batch_size)
  local valsamples = valrows * batch_size - n
  local valid_input_windows = torch.LongTensor(valsamples, n)
  local valid_output_windows = torch.DoubleTensor(valsamples)
  for i = 1, valsamples do
    valid_input_windows[i] = valid_input_all[{{ i, i + n - 1 }}]
    valid_output_windows[i] = valid_output_all[i + n - 1]
  end

	-- Initialize network
	local net = nn.Sequential()
	net:add(nn.LookupTable(nfeatures, din)):add(nn.Reshape(n*din, true)):add(nn.Linear(n*din, dhid)):add(nn.Tanh()):add(nn.Linear(dhid, 1)):add(nn.Sigmoid())
	local criterion = nn.BCECriterion()

  -- Get parameters
  params, gradParams = net:getParameters()
  for i = 1, params:size()[1] do
    params[i] = torch.uniform(-0.05, 0.05)
  end
  local prevLoss = 10000

	-- Train network
	for t = 1, epochs do
		print(t)
		-- Create minibatches
		local train_input_mb = torch.LongTensor(bsize, n)
		local train_output_mb = torch.LongTensor(bsize)

		for j = 1, torch.floor(nsamples / bsize) + 1 do
			local train_input_mb = input_windows[{{ (j-1) * bsize + 1, math.min(j * bsize, nsamples) }}]
			local train_output_mb = output_windows[{{ (j-1) * bsize + 1, math.min(j * bsize, nsamples) }}]

			-- Manual SGD
			criterion:forward(net:forward(train_input_mb), train_output_mb)
			net:zeroGradParameters()
			net:backward(train_input_mb, criterion:backward(net.output, train_output_mb))
			net:updateParameters(lambda)
		end

    -- Accuracy on development set
    local val_pred = net:forward(valid_input_windows)
    local currLoss = criterion:forward(val_pred, valid_output_windows)
    print("NNLM validation loss: " .. currLoss)
    local val_perp = perplexity(val_pred, valid_output_windows, 'nnlm')
    print("NNLM validation perplexity: " .. val_perp)
    lambda = adaptiveLambda(lambda, currLoss, prevLoss, eps)
    prevLoss = currLoss
	end

	return net:forward(valid_input_windows)
end

function RNN(train_input, train_output, clf, din, dhid, epochs, seqlen, lambda, eps, dropout)
  -- Transpose input for RNN
  local train_input_T = train_input:t() -- batch_size x num_batch
  local train_output_T = train_output:t()
  local batch_size, nbatch = train_input_T:size()[1], train_input_T:size()[2]
  local valid_input_T = valid_input:t()
  local valid_output_T = valid_output:t()
  local valid_output_table = {}
  for i = 1, batch_size do
    valid_output_table[i] = valid_output_T[i]
  end
  local valbatch = valid_input_T:size()[2]

  -- Initialize network (LSTM/RNN/GRU)
  local net = nn.Sequential()
  local LT = nn.LookupTable(nfeatures, din) -- batch_size x nbatch x din
  net:add(LT)
  net:add(nn.SplitTable(1, 3)) -- batch_size table of nbatch x din
  if clf == 'lstm' then
    local seqLSTM = nn.Sequencer(nn.Sequential()
      :add(nn.LSTM(din, 1))
      :add(nn.Sigmoid())
    ) -- batch_size table of nbatch x 1
    seqLSTM:remember('both')
    net:add(seqLSTM)
  elseif clf == 'rnn' then
    local rm = nn.Sequential() -- x is nbatch x din; s is nbatch x dhid
    rm:add(nn.JoinTable(2)) -- [x,s] is nbatch x (din + dhid)
    rm:add(nn.Linear(din + dhid, dhid)) -- [x,s]*W + b yields nbatch x dhid s'
    rm:add(nn.Tanh()) -- tanh([x,s] * W + b)
    local seqRNN = nn.Sequencer(nn.Sequential()
      :add(nn.Recurrence(rm, dhid, 1))
      :add(nn.Linear(dhid, 1))
      :add(nn.Sigmoid())
    )
    seqRNN:remember('both')
    net:add(seqRNN)
  else -- 'gru'
    local seqGRU = nn.Sequencer(nn.Sequential()
      :add(nn.GRU(din, 1, 99999, dropout))
      :add(nn.Sigmoid())
    )
    seqGRU:remember('both')
    net:add(seqGRU)
  end
  local criterion = nn.SequencerCriterion(nn.BCECriterion())

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
    local val_perp = perplexity(val_pred, valid_output_T, 'rnn')
    print("RNN validation perplexity: " .. val_perp)
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
   nfeatures = f:read('nfeatures'):all():long()[1]
   train_input = f:read('train_input'):all():double()
   train_output = f:read('train_output'):all():double() - 1
   valid_input = f:read('valid_input'):all():double()
   valid_output = f:read('valid_output'):all():double() - 1
   seqlen = f:read('seqlen'):all():long()[1]
   batch_size = f:read('bsize'):all():long()[1]

   -- Parse Hyperparameters
   local clf = opt.classifier
   local n = opt.n
   local smooth = opt.smooth
   local din = opt.din
   local dhid = opt.dhid
   local epochs = opt.epochs
   local bsize = opt.bsize
   local seqlen = opt.seqlen
   local lambda = opt.lambda
   local eps = opt.eps
   local dropout = opt.dropout

   -- Train.
   if clf == 'count' then
     predictions = CountBased(train_input, train_output, n, smooth)
   elseif clf == 'nnlm' then
     predictions = NNLM(train_input, train_output, n, din, dhid, epochs, bsize, lambda, eps)
   else
     predictions = RNN(train_input, train_output, clf, din, dhid, epochs, seqlen, lambda, eps, dropout)
   end

   -- Test.
end

main()