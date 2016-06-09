-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
-- ...

function evaluate(W, b, input, output)
	predict = torch.round(input * W + b)

	correct = 0
	n = output:size()[1]
	for i = 1,n do
		if predict[i] == output[i] then
			correct = correct + 1
		end
	end

	return correct / n
end

function NB(train_input, train_output, W, b, alpha)
	local nclasses, nfeatures = W:size()[1], W:size()[2]
	local n, maxSentLen = train_input:size()[1], train_input:size()[2] -- n is sample size
	
	-- Fill in W, b tensors with counts
	for i = 1, n do
		c = train_output[i] -- Current class
		b[c] = b[c] + 1
		for j = 1, maxSentLen do
			f = train_input[i][j] - 1 -- to account for padding '1'
			if f > 0 then
				W[c][f] = W[c][f] + 1
			end
		end
	end

	-- Normalize to get probabilities
	for c = 1, nclasses do
		b[c] = (b[c] + alpha) / (n + alpha * nclasses)
		cTotal = W[{ {}, c }]:sum()
		for f = 1, nfeatures do
			W[c][f] = (W[c][f] + alpha) / (cTotal + alpha * nfeatures)
		end
	end

	-- Return log probabilities
	b = torch.log(b)
	W = torch.log(W)
end

function LR(train_input, train_output, W, b, batchSize, stepSize, epochs, lambda)
	-- W, b are the initial values; epsilon is convergence parameter
	local nclasses, nfeatures = W:size()[1], W:size()[2]
	local n, maxSentLen = train_input:size()[1], train_input:size()[2]

	-- Initialize losses/gradients/predictions
	local gradW = torch.DoubleTensor(nclasses, nfeatures)
	local gradb = torch.DoubleTensor(nclasses)
	local yHat = torch.DoubleTensor(nclasses) -- predictions

	for t = 1, epochs do
		sample = torch.randperm(n)[{{1,m}}] -- sample m random examples

		-- Reinitialize with current regularized gradient values
		gradW = gradW:zero() + lambda * W
		gradb = gradb:zero() + lambda * b
		yHat:zero()

		for i = 1, n do
			if sample:eq(i):sum() > 0 then -- check if index in sample;

				-- Compute yHat_c in sample (each yHat is just class probabilities for given i sample); not normalized
				for c in 1, nclasses do
					res = 0
					for w in 1, maxSentLen do
						if train_input[i][w] > 1 then -- check not padding
							res = res + W[c][train_input[i][w]-1]
						end
					end
					yHat[c] = res + b[c]
				end

				-- Compute log-sum-exp
				local M = yHat:max()
				local partition = torch.log(torch.exp(yHat - M):sum()) + M
				yHat = torch.exp(yHat - partition)

				for c in 1, nclasses do
					-- Compute incremental gradients
					if train_output[i] == c then
						gradb[c] = gradb[c] - (1/m) * (1 - yHat[c])
						for f in 1, nfeatures do
							gradW[{c,f}] = gradW[{c,f}] - (1/m) * train_input[i]:eq(f + 1):sum() * (1 - yHat[c])
						end
					else
						gradb[c] = gradb[c] + (1/m) * yHat[c]
						for f in 1, nfeatures do
							gradW[{c,f}] = gradW[{c,f}] - (1/m) * train_input[i]:eq(f + 1):sum() * yHat[c]
						end
					end
				end
			end
		end

		-- Update W, b
		W = W - stepSize * gradW
		b = b - stepSize * gradb

	end
end

function SVM(train_input, train_output, W, b, batchSize, stepSize epochs, lambda)

	local nclasses, nfeatures = W:size()[1], W:size()[2]
	local n, maxSentLen = train_input:size()[1], train_input:size()[2]

	-- Initialize losses/gradients/predictions
	local gradW = torch.DoubleTensor(nclasses, nfeatures)
	local gradb = torch.DoubleTensor(nclasses)
	local yHat = torch.DoubleTensor(nclasses) -- predictions

	for t = 1, epochs do
		sample = torch.randperm(n)[{{1,m}}] -- sample m random examples

		-- Reinitialize with current regularized gradient values
		gradW = gradW:zero() + lambda * W
		gradb = gradb:zero() + lambda * b
		yHat:zero()

		for i = 1, n do
			if sample:eq(i):sum() > 0 then -- check if index in sample;
				
				-- Compute yHat_c in sample (each yHat is just class probabilities for given i sample); not normalized
				for c in 1, nclasses do
					res = 0
					for w in 1, maxSentLen do
						if train_input[i][w] > 1 then -- check not padding
							res = res + W[c][train_input[i][w]-1]
						end
					end
					yHat[c] = res + b[c]
				end

				-- Compute log-sum-exp
				local M = yHat:max()
				local partition = torch.log(torch.exp(yHat - M):sum()) + M
				yHat = torch.exp(yHat - partition)

				-- Find c and c'
				local trueClass = train_output[i]
				local sorted, indices = yHat:sort() -- sort into ascending order, so take nclasses-1 to find c'
				local highClass = indices[nclasses - 1]

				-- Check if |yHat_c - yHat_c'| > 1
				if torch.abs(yHat[trueClass] - yHat[highClass]) <= 1 then
					
					-- Update gradients incrementally
					for f in 1, nfeatures do
						gradW[{trueClass,f}] = gradW[{trueClass,f}] - (1/m) * train_input[i]:eq(f + 1):sum() * ( -yHat[highClass]*yHat[trueClass] - yHat[trueClass]*(1 - yHat[trueClass]) )
						gradW[{highClass,f}] = gradW[{highClass,f}] - (1/m) * train_input[i]:eq(f + 1):sum() * ( yHat[highClass]*(1 - yHat[highClass]) + yHat[trueClass]*yHat[highClass] )
					end

					gradb[trueClass] = gradb[trueClass] - (1/m) * ( -yHat[highClass]*yHat[trueClass] - yHat[trueClass]*(1 - yHat[trueClass]) )
					gradb[highClass] = gradb[highClass] - (1/m) * ( yHat[highClass]*(1 - yHat[highClass]) + yHat[trueClass]*yHat[highClass] )
				end
			end
		end

		-- Update W, b
		W = W - stepSize * gradW
		b = b - stepSize * gradb

	end
end


function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]

   local W = torch.DoubleTensor(nclasses, nfeatures)
   local b = torch.DoubleTensor(nclasses)

   -- Train.

   -- Test.
end

--main()
