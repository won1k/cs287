-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-lm', 'kn', 'classifier to use')
cmd:option('-smooth', 0, 'smoothing parameter (alpha if clf = Laplace, D if clf = kn')

-- Hyperparameters
cmd:option('-k', 10, 'number of noise samples in NCE')

-- Utility functions
function wb(F, i, indices)
	local n = indices:size()
	if i > 1 then
		local Fsum = F[torch.totable(indices[{{n-i+1,n-1}}])]:sum()
		local Nsum = F[torch.totable(indices[{{n-i+1,n-1}}])]:gt(0):sum()]
		return (F[torch.totable(indices[{{n-i+1,n}}])] + Nsum * wb(F, i-1, indices)) / (Fsum + Nsum)
	else
		return F[torch.totable(indices[{{n}}])] / F:sum()
end

function normalize(F, n, clf, D)
	indices[n] = indices[n] + 1
	for i = n, 1, -1 do
		if indices[i] > nclasses then
			indices[i-1] = indices[i-1] + 1
			indices[i] = 1
		end
	end
	print(indices)
	indexSum = F[torch.totable(indices[{{1,n-1}}])]:sum()
	if indexSum > 0 then
		if clf == 'laplace' or clf == 'ml' then 
			return function() return F[torch.totable(indices)] / indexSum end
		elseif clf == 'kn' then
			return function() return (math.max(F[torch.totable(indices)] - D, 0) + D * F[torch.totable(indices[{{1,n-1}}])]:gt(0):sum() * F[torch.totable(indices[{{2,n}}])]:gt(0):sum()) / F[torch.totable(indices[{{2,n-1}}])]:gt(0):sum()) / indexSum end
		else
			return function() return wb(F, n, n, indices) end
		end
	else
		return function() return 0 end
	end
end


-- Models
function CountBased(train_input, train_output, clf, alpha, D)
	nclasses, n = W:size()[1], W:size():size()
	nsamples, contextSize = train_input:size()[1], train_input:size()[2]
	-- Fill in F tensor with counts
	F = torch.DoubleTensor(W:size()):fill(alpha) -- add alpha to all counts if > 0
	for i = 1, nsamples do
		local c = train_input[i] -- context
		local w = train_output[i]
		local idx = torch.totable(c)
		F[idx][w] = F[idx][w] + 1
	end
	-- Normalize to get probabilities / smoothing
	indices = torch.LongTensor(n):fill(1)
	indices[n] = 0
	W:apply(normalize(F, clf, D))
end

function NNLM()
end

function NCE_manual(train_input, train_output, din, dhid, epochs, bsize, k)
	-- Compute unigram probabilities for noise distribution
	local unigram_probs = torch.LongTensor(nclasses)
	for i = 1, nsamples do unigram_probs[train_output[i]] = unigram_probs[train_output[i]] + 1 end
	local totalCounts = unigram_probs:sum()
	unigram_probs:apply(function(x) return(x / totalCounts) end)

	-- Initialize network
	local net = nn.Sequential()
	net:add(nn.LookupTable(nclasses, din)):add(nn.Reshape(din * dwin, true)):add(nn.Linear(din * dwin, dhid)):add(nn.Tanh()):add(nn.Linear(dhid, nclasses))

	-- Train network
	for t = 1, epochs do
		-- Create minibatches
		local train_input_mb = torch.LongTensor(bsize)
		local train_output_mb = torch.DoubleTensor(bsize)
		local mb_idx = torch.randperm(nsamples)[{{1,bsize}}] -- sample batch-size random examples

		for i = 1, bsize do
			train_input[mb_idx[i]]
			train_output[mb_idx[i]]

			-- Generate noise samples
			local noiseIndices = torch.randperm(nsamples)[{{1,k}}]
			local noiseSamples = torch.LongTensor(k)
			for j = 1, k do noiseSamples[j] = train_output[noiseIndices[j]] end

			-- Manual SGD
			local pred = net:forward(train_input[mb_idx[i]]) -- Predictions
			-- Compute derivative of loss wrt output
			local lossDerivs = 
			net:zeroGradParameters()
			net:backward(train_input, lossDerivs)
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

end

function NCE_network(train_input, train_output, din, dhid, epochs, bsize, k)
	-- Compute unigram probabilities for NCE computation
	local unigram_probs = torch.LongTensor(nclasses)
	for i = 1, nsamples do unigram_probs[train_output[i]] = unigram_probs[train_output[i]] + 1 end
	local totalCounts = unigram_probs:sum()
	unigram_probs:apply(function(x) return(x / totalCounts) end)

	-- Initialize network
	local net = nn.Sequential()
	net:add(nn.LookupTable(nclasses, din)):add(nn.Reshape(din * dwin, true)):add(nn.Linear(din * dwin, dhid)):add(nn.Tanh())
	local linear = nn.Linear(dhid, nclasses)
	net:add(linear)

	-- Initialize NCE network
	local nce = nn.Sequential()
	nce:add(nn.LookupTable(nclasses, din)):add(nn.Reshape(din * dwin, true)):add(nn.Linear(din * dwin, dhid)):add(nn.Tanh())
	local sublinear = nn.Linear(dhid, 32 + k)
	nce:add(sublinear)

	-- Initialize loss criterion network
	local loss_net = nn.Sequential()
	local add_k_prob = nn.Add(bsize + k, true)
	loss_net:add(add_k_prob):add(nn.Sigmoid())
	

	-- Train network
	for t = 1, epochs do
		-- Create minibatches
		local train_input_mb = torch.LongTensor(bsize)
		local train_output_mb = torch.LongTensor(bsize)
		local mb_idx = torch.randperm(nsamples)[{{1,bsize}}]

		-- Generate noise samples
		local noise_indices = torch.randperm(nsamples)[{{1,k}}]
		local noise_samples = torch.LongTensor(k)
		for j = 1, k do noise_samples[j] = train_output[noise_indices[j]] end

		-- Transfer weights
		for i = 1, bsize do
			train_input_mb[i] = train_input[mb_idx[i]]
			train_output_mb[i] = train_output[mb_idx[i]]
			sublinear.weight[i] = linear.weight[train_output_mb[i]]
			sublinear.bias[i] = linear.bias[train_output_mb[i]]
		end
		for i = 1, k do
			sublinear.weight[bsize + i] = linear.weight[noise_samples[i]]
			sublinear.bias[bsize + i] = linear.bias[noise_samples[i]]
		end

		-- Predictions
		local pred = nce:forward(train_input_mb)

		-- Compute sigmoid functions for NCE; fill into NLL; compute loss derivatives
		local derivs = torch.DoubleTensor(bsize + k):fill(0)
		local all_samples = torch.cat(train_output_mb, noise_samples)
		for i = 1, bsize do
			pred[i]:map(all_samples, function(z, w) 
				return torch.sigmoid(z - torch.log(k*unigram_probs[w]))
			end)
			local criterion = nn.ClassNLLCriterion()
			criterion:forward(pred[i], all_samples)
			derivs = derivs + (1/bsize) * criterion:backward(pred[i], all_samples)
		end

		-- Backpropagate
		nce:backward(train_input_mb, derivs)

		-- Transfer weights back to main NN
		for i = 1, bsize do
			linear.weight[train_output_mb[i]] = sublinear.weight[i]
			linear.bias[train_output_mb[i]] = sublinear.bias[i]
		end
		for i = 1, k do
			linear.weight[noise_samples[i]] = sublinear.weight[bsize + i]
			linear.bias[noise_samples[i]] = sublinear.bias[bsize + i]
		end
	end

end


function main() 
   	-- Parse input params
   	opt = cmd:parse(arg)
	local f = hdf5.open(opt.datafile, 'r')
	local nclasses = f:read('nclasses'):all():long()[1]
	local n = f:read('n'):all():long()[1]
	local clf = opt.lm
	local alpha = opt.alpha
	local D = opt.D

	local dims = torch.LongStorage(n):fill(nclasses)
	W = torch.DoubleTensor(dims)

   	-- Load training data
    local train_input = f:read('train_input'):all():long()
    local train_output = f:read('train_output'):all():long()

    -- Load development data
    valid_input_word_windows = f:read('valid_input_word_windows'):all():long()
    valid_input_cap_windows = f:read('valid_input_cap_windows'):all():long()
    valid_input = torch.DoubleTensor(valid_input_word_windows:size()[1],dwin,2)
	valid_input[{{},{},1}] = valid_input_word_windows
	valid_input[{{},{},2}] = valid_input_cap_windows
	valid_output = f:read('valid_output'):all():long()

    -- Train.
    if clf == 'laplace' or clf == 'ml' or clf == 'kn' then
   		CountBased(train_input, train_output, clf, alpha, D)
   	elseif clf == 'nce' then
   		local din = 30
   		local dhid = 100
   		local epochs = 20
   		local bsize = 32
   		local k = 10
   		NCE_network(train_input, train_output, din, dhid, epochs, bsize, k)
   	end

    -- Test.
end

main()
