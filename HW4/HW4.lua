-- Only requirement allowed
require("hdf5")
require("rnn")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
cmd:option('-n', 3, 'n-gram parameter')
cmd:option('-smooth', 0, 'smoothing parameter (alpha if clf = Laplace, D if clf = kn')
cmd:option('-din', 15, 'dimension of word embedding')
cmd:option('-dhid', 100, 'dimension of hidden layer')
cmd:option('-lambda', 0.1, 'learning rate')
cmd:option('-epochs', 20, 'number of training epochs')
cmd:option('-bsize', 32, 'batch size')

-- Models

function NNLM(train_input, train_output, n, din, dhid, epochs, bsize, lambda)
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

	-- Initialize network
	local net = nn.Sequential()
	net:add(nn.LookupTable(nfeatures, din)):add(nn.Reshape(n*din, true)):add(nn.Linear(n*din, dhid)):add(nn.Tanh()):add(nn.Linear(dhid, 2)):add(nn.LogSoftMax())
	local criterion = nn.ClassNLLCriterion()

	-- SGD training
	for t = 1, epochs do
		print(t)
		-- Create minibatches
		local train_input_mb = torch.LongTensor(bsize, n)
		local train_output_mb = torch.LongTensor(bsize)
		local mb_idx = torch.randperm(nsamples) -- randomly permute indices

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
    local values, predictions = torch.max(net:forward(valid_input_windows), 2)
    local accuracy = 0
    for i = 1, valsamples do
      if predictions[i][1] == valid_output_windows[i] then
        accuracy = accuracy + 1
      end
    end
    print(accuracy / valsamples)
	end
	return net:forward(valid_input_windows)
end

function main()
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   nfeatures = f:read('nfeatures'):all():long()[1]
   train_input = f:read('train_input'):all():long()
   train_output = f:read('train_output'):all():long()
   valid_input = f:read('valid_input'):all():long()
   valid_output = f:read('valid_output'):all():long()
   seqlen = f:read('seqlen'):all():long()[1]
   batch_size = f:read('bsize'):all():long()[1]

   -- Parse Hyperparameters
   local n = opt.n
   local din = opt.din
   local dhid = opt.dhid
   local epochs = opt.epochs
   local bsize = opt.bsize
   local lambda = opt.lambda

   -- Train.
   predictions = NNLM(train_input, train_output, n, din, dhid, epochs, bsize, lambda)

   -- Test.
end

main()
