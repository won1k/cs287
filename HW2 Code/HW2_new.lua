-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")
require("prelookup") -- Custom PreLookupTable for pretrained features

-- Global constants
ncap = 6 -- number of cap features
dpre = 50 -- dimension of dense word features (pretrained)

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'PTB.hdf5', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
cmd:option('-alpha', 0, 'alpha')
cmd:option('-dword', 50, 'word dimension')
cmd:option('-dcaps', 5, 'caps dimension')
cmd:option('-lambda', 0.1, 'learning rate')
cmd:option('-epochs', 100, 'epochs')
cmd:option('-bsize', 32, 'mini-batch size')
cmd:option('-dhid', 300, 'hidden dimension')
cmd:option('-pretrained', 0, 'pretrained features')

-- LookupTable with pretrained features
function PreLookup(train_input_word_windows, pretrained_features)
	local n, dwin = train_input_word_windows:size()[1], train_input_word_windows:size()[2]
	local dense_word_windows = torch.DoubleTensor(n, dwin, dpre)
	for i = 1, n do
		for j = 1, dwin do
			dense_word_windows[i][j] = pretrained_features[train_input_word_windows[i][j]]
		end
	end
	return dense_word_windows
end

-- Naive Bayes implementation
function NB(train_input_word_windows, train_input_cap_windows, train_output, W, b, alpha)
	local nclasses, nfeatures = W:size()[1], W:size()[2]
	local n, dwin = train_input_word_windows:size()[1], train_input_word_windows:size()[2]
	nfeatures = nfeatures / dwin
	
	-- Fill in W, b tensors with counts
	for i = 1, n do
		c = train_output[i] -- Current class
		b[c] = b[c] + 1
		for j = 1, dwin do
			-- Word feature
			f = train_input_word_windows[i][j]
			W[c][(j-1)*nfeatures + f] = W[c][(j-1)*nfeatures + f] + 1
			-- Caps feature
			fc = train_input_cap_windows[i][j]
			if fc > 0 then
				W[c][j*nfeatures - ncap + fc] = W[c][j*nfeatures - ncap + fc] + 1
			end
		end
	end

	-- Normalize to get probabilities
	for c = 1, nclasses do
		b[c] = (b[c] + alpha) / (n + alpha * nclasses)
		cTotal = W[{ {}, c }]:sum()
		for f = 1, nfeatures * dwin do
			W[c][f] = (W[c][f] + alpha) / (cTotal + alpha * nfeatures * dwin)
		end
	end

	-- Return log probabilities
	b = torch.log(b)
	W = torch.log(W)
end

-- Multiclass logistic regression implementation
function LR(train_input_word_windows, train_input_cap_windows, train_output, alpha, dword, dcaps, lambda, epochs, bsize)
	local n, dwin = train_input_word_windows:size()[1], train_input_word_windows:size()[2]

	-- Concatenate word/cap data
	local train_input = torch.DoubleTensor(n,dwin,2)
	train_input[{{},{},1}] = train_input_word_windows
	train_input[{{},{},2}] = train_input_cap_windows

	-- Initialize network
	local mlr = nn.Sequential()
	local c = nn.Concat(2)
	local wrd = nn.Sequential()
	wrd:add(nn.Select(3,1))
	wrd:add(nn.LookupTable(nfeatures - ncap, dword))
	wrd:add(nn.Reshape(dword * dwin, true))
	c:add(wrd)
	local caps = nn.Sequential()
	caps:add(nn.Select(3,2))
	caps:add(nn.LookupTable(ncap, dcaps))
	caps:add(nn.Reshape(dcaps * dwin, true))
	c:add(caps)
	mlr:add(c)
	mlr:add(nn.Linear((dword + dcaps) * dwin, nclasses))
	mlr:add(nn.LogSoftMax())

	-- Define criterion, initialize parameters
	local criterion = nn.ClassNLLCriterion()
	local params, gradParams = mlr:getParameters()
	local optimState = {learningRate = lambda}

	-- Train network
	for t = 1, epochs do
		-- Create minibatches
		local train_input_mb = torch.DoubleTensor(bsize, dwin, 2)
		local train_output_mb = torch.DoubleTensor(bsize)
		mb_idx = torch.randperm(n)[{{1,bsize*1000}}] -- sample batch-size random examples

		for j = 1, 1000 do
			for i = 1, bsize do
				train_input_mb[i] = train_input[mb_idx[(j-1)*bsize + i]]
				train_output_mb[i] = train_output[mb_idx[(j-1)*bsize + i]]
			end

			-- Evaluation function for SGD
			--local function eval(params)
			--	gradParams:zero()

			--	local predictions = mlr:forward(train_input_mb)
			--	local loss = criterion:forward(predictions, train_output_mb)
			--	local gradLoss = criterion:backward(predictions, train_output_mb)
			--	mlr:backward(train_input_mb, gradLoss)

			--	return loss, gradParams
			--end

			-- Manual SGD
			criterion:forward(mlr:forward(train_input_mb), train_output_mb)
			mlr:zeroGradParameters()
			mlr:backward(train_input_mb, criterion:backward(mlr.output, train_output_mb))
			mlr:updateParameters(lambda)

			-- Compute average loss at each step
			--local currLoss, currGrad = eval(params)
			--print("The current loss is: " .. currLoss)

			-- Perform SGD
			-- optim.sgd(eval, params, optimState)
		end

		-- Compute performance on development set
		devPred = mlr:forward(valid_input)
		devLoss = criterion:forward(devPred, valid_output)
		print("The loss on the validation set is: " .. devLoss)
		maxPred, maxIdx = torch.max(devPred, 2)
		devAcc = torch.eq(maxIdx, valid_output):sum() / valid_output:size()[1]
		print("The accuracy on the validation set is: " .. devAcc)
	end

	return 
end

-- Neural network model (no pretrained vectors)
function NN(train_input_word_windows, train_input_cap_windows, train_output, alpha, dword, dcaps, lambda, epochs, bsize, dhid)
	local n, dwin = train_input_word_windows:size()[1], train_input_word_windows:size()[2]

	-- Concatenate word/cap data
	local train_input = torch.DoubleTensor(n,dwin,2)
	train_input[{{},{},1}] = train_input_word_windows
	train_input[{{},{},2}] = train_input_cap_windows

	-- Initialize network
	local net = nn.Sequential()
	local c = nn.Concat(2)
	local wrd = nn.Sequential()
	wrd:add(nn.Select(3,1))
	wrd:add(nn.LookupTable(nfeatures - ncap, dword))
	wrd:add(nn.Reshape(dword * dwin, true))
	c:add(wrd)
	local caps = nn.Sequential()
	caps:add(nn.Select(3,2))
	caps:add(nn.LookupTable(ncap, dcaps))
	caps:add(nn.Reshape(dcaps * dwin, true))
	c:add(caps)
	net:add(c)
	net:add(nn.Linear((dword + dcaps) * dwin, dhid))
	net:add(nn.HardTanh())
	net:add(nn.Linear(dhid, nclasses))
	net:add(nn.LogSoftMax())

	-- Define criterion, initialize parameters
	local criterion = nn.ClassNLLCriterion()

	-- Train network
	for t = 1, epochs do
		-- Create minibatches
		local train_input_mb = torch.DoubleTensor(bsize, dwin, 2)
		local train_output_mb = torch.DoubleTensor(bsize)
		mb_idx = torch.randperm(n)[{{1,bsize*1000}}] -- sample batch-size random examples

		for j = 1, 1000 do
			for i = 1, bsize do
				train_input_mb[i] = train_input[mb_idx[(j-1)*bsize + i]]
				train_output_mb[i] = train_output[mb_idx[(j-1)*bsize + i]]
			end

			-- Manual SGD
			criterion:forward(net:forward(train_input_mb), train_output_mb)
			net:zeroGradParameters()
			net:backward(train_input_mb, criterion:backward(net.output, train_output_mb))
			net:updateParameters(lambda)
		end

		-- Compute performance on development set
		devPred = net:forward(valid_input)
		devLoss = criterion:forward(devPred, valid_output)
		print("The loss on the validation set is: " .. devLoss)
		maxPred, maxIdx = torch.max(devPred, 2)
		devAcc = torch.eq(maxIdx, valid_output):sum() / valid_output:size()[1]
		print("The accuracy on the validation set is: " .. devAcc)
	end

	return 
end

-- Neural network model (pretrained vectors)
function NNPre(train_input_word_windows, train_input_cap_windows, train_output, pretrained_features, alpha, dword, dcaps, lambda, epochs, bsize, dhid)
	local n, dwin = train_input_word_windows:size()[1], train_input_word_windows:size()[2]

	-- Prepare data for network
	local train_input = { PreLookup(train_input_word_windows, pretrained_features), train_input_cap_windows }
	local valid_input = { PreLookup(valid_input_word_windows, pretrained_features), valid_input_cap_windows }

	-- Initialize network
	local net = nn.Sequential()
	local c = nn.ParallelTable()
	local wrd = nn.Reshape(dpre * dwin, true)
	c:add(wrd)
	local caps = nn.Sequential()
	caps:add(nn.LookupTable(ncap, dcaps))
	caps:add(nn.Reshape(dcaps * dwin, true))
	c:add(caps)
	net:add(c)
	net:add(nn.JoinTable(2))
	net:add(nn.Linear((dpre + dcaps) * dwin, dhid))
	net:add(nn.HardTanh())
	net:add(nn.Linear(dhid, nclasses))
	net:add(nn.LogSoftMax())

	-- Define criterion, initialize parameters
	local criterion = nn.ClassNLLCriterion()

	-- Train network
	for t = 1, epochs do
		-- Create minibatches
		local train_input_mb = {torch.DoubleTensor(bsize, dwin, dpre), torch.DoubleTensor(bsize, dwin)}
		local train_output_mb = torch.DoubleTensor(bsize)
		mb_idx = torch.randperm(n)[{{1,bsize*1000}}] -- sample batch-size random examples

		for j = 1, 1000 do
			for i = 1, bsize do
				train_input_mb[1][i] = train_input[1][mb_idx[(j-1)*bsize + i]]
				train_input_mb[2][i] = train_input[2][mb_idx[(j-1)*bsize + i]]
				train_output_mb[i] = train_output[mb_idx[(j-1)*bsize + i]]
			end

			-- Manual SGD
			criterion:forward(net:forward(train_input_mb), train_output_mb)
			net:zeroGradParameters()
			net:backward(train_input_mb, criterion:backward(net.output, train_output_mb))
			net:updateParameters(lambda)
		end

		-- Compute performance on development set
		devPred = net:forward(valid_input)
		devLoss = criterion:forward(devPred, valid_output)
		print("The loss on the validation set is: " .. devLoss)
		maxPred, maxIdx = torch.max(devPred, 2)
		devAcc = torch.eq(maxIdx, valid_output):sum() / valid_output:size()[1]
		print("The accuracy on the validation set is: " .. devAcc)
	end

	return
end

function main() 
   	-- Parse input params
   	opt = cmd:parse(arg)
   	local f = hdf5.open(opt.datafile, 'r')
    nclasses = f:read('nclasses'):all():long()[1]
    nfeatures = f:read('nfeatures'):all():long()[1]
    dwin = f:read('dwin'):all():long()[1]

    -- Parse hyperparameters
    local clf = opt.classifier
    local alpha = opt.alpha
    local dword = opt.dword
    local dcaps = opt.dcaps
    local lambda = opt.lambda
    local epochs = opt.epochs
    local bsize = opt.bsize
    local dhid = opt.dhid
    local pretrained = opt.pretrained
    -- ...

    -- Initialize weight tensors
    if opt.classifier == 'nb' then
    	local W = torch.DoubleTensor(nclasses, nfeatures * dwin) -- i.e. word at i in x, so indexed like: [w1@1 w2@1 ... c1@1 c2@1 ... w1@2 w2@2 ... ...... ... w1@dwin w2@dwin ...]
    else
    	local W = torch.DoubleTensor(nclasses, nfeatures)
    end
    local b = torch.DoubleTensor(nclasses)

    -- Load training data
    local train_input_word_windows = f:read('train_input_word_windows'):all():long()
    local train_input_cap_windows = f:read('train_input_cap_windows'):all():long()
    local train_output = f:read('train_output'):all():long()
    if pretrained > 0 then
    	pretrained_features = f:read('pretrained_features'):all():double()
    end

    -- Load development data
    valid_input_word_windows = f:read('valid_input_word_windows'):all():long()
    valid_input_cap_windows = f:read('valid_input_cap_windows'):all():long()
    valid_input = torch.DoubleTensor(valid_input_word_windows:size()[1],dwin,2)
	valid_input[{{},{},1}] = valid_input_word_windows
	valid_input[{{},{},2}] = valid_input_cap_windows
	valid_output = f:read('valid_output'):all():long()
 

    -- Train.
    if clf == 'nb' then
    	NB(train_input_word_windows, train_input_cap_windows, train_output, W, b, alpha)
    elseif clf == 'lr' then
    	LR(train_input_word_windows, train_input_cap_windows, train_output, alpha, dword, dcaps, lambda, epochs, bsize)
    elseif clf == 'nn' then
    	NN(train_input_word_windows, train_input_cap_windows, train_output, alpha, dword, dcaps, lambda, epochs, bsize, dhid)
    else
    	NNPre(train_input_word_windows, train_input_cap_windows, train_output, pretrained_features, alpha, dword, dcaps, lambda, epochs, bsize, dhid)
    end

    -- Test.
end

main()
