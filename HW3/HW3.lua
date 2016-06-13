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
-- ...

-- Utility functions
function contToIdx(c, nclasses, contextSize)
	-- All indices besides (i-1) start at 0; (i-1) starts at 1
	local idx = 0
	for i = 1, contextSize do
		if i < contextSize then
			idx = idx + (c[i]-1) * nclasses^(contextSize - i)
		else
			idx = idx + c[i]
		end
	end
	return idx
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
	if clf == 'laplace' or clf == 'ml' then 
		if indexSum > 0 then
			return function() return F[torch.totable(indices)] / indexSum end
		else
			return function() return 0 end
		end
	elseif clf == 'kn' then
		if indexSum > 0 then
			return function() return (math.max(F[torch.totable(indices)] - D, 0) + D * F[torch.totable(indices[{{1,n-1}}])]:gt(0):sum() * F[torch.totable(indices[{{2,n}}])]:gt(0):sum()) / F[torch.totable(indices[{{2,n-1}}])]:gt(0):sum()) / indexSum end
		else
			return function() return 0 end
		end
	else
		if indexSum > 0 then
			return function() return F[torch.totable(indices) + F[torch.totable(indices[{{1,n-1}}])]:gt(0):sum()] * 
	end
end


-- Models
function CountBased(train_input, train_output, clf, alpha, D)
	nclasses, n = W:size()[1], W:size():size()
	samples, contextSize = train_input:size()[1], train_input:size()[2]
	-- Fill in F tensor with counts
	F = torch.DoubleTensor(W:size()):fill(alpha) -- add alpha to all counts if > 0
	for i = 1, samples do
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

function KneserNey(train_input, train_output)

function NNLM()
end

function NCE()
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
   	elseif clf == 'kn'

    -- Test.
end

main()
