local THNN = require 'nn.THNN'
local nn = require 'nn'
local hdf5 = require 'hdf5'
local PreLookupTable, parent = torch.class('PreLookupTable', 'nn.Module')

PreLookupTable.__version = 4

function PreLookupTable:__init()
   parent.__init(self)

   local f = hdf5.open("PTB55.hdf5", "r")
   self.weight = f:read('pretrained_features'):all():double()
   self.gradWeight = torch.Tensor(self.weight:size()[1], 50):zero()
   self.paddingValue = 0
   self.maxNorm = nil
   self.normType = nil

   self:reset()
end

function PreLookupTable:backCompatibility()
   self._count = self._count or torch.IntTensor()
   self._input = self._input or torch.LongTensor()

   if not self.shouldScaleGradByFreq then
      self.shouldScaleGradByFreq = false
   end
end

function PreLookupTable:accUpdateOnly()
   self.gradWeight = nil
   return self
end

function PreLookupTable:setPadding(paddingValue)
   self.paddingValue = paddingValue
   return self
end

function PreLookupTable:setMaxNorm(maxNorm)
   self.maxNorm = maxNorm
   return self
end

function PreLookupTable:setNormType(normType)
   self.normType = normType
   return self
end

function PreLookupTable:scaleGradByFreq()
   self.shouldScaleGradByFreq = true
   return self
end

function PreLookupTable:reset(stdv)
   stdv = stdv or 1
end

function PreLookupTable:makeInputContiguous(input)
   -- make sure input is a contiguous torch.LongTensor
   if (not input:isContiguous()) or torch.type(input) ~= torch.type(self._input) then
      self.copiedInput = true
      self._input:resize(input:size()):copy(input)
      return self._input
   end
   self.copiedInput = false
   return input
end

function PreLookupTable:updateOutput(input)
   self:backCompatibility()
   self:renorm(input)
   input = self:makeInputContiguous(input)
   if input:dim() == 1 then
      self.output:index(self.weight, 1, input)
   elseif input:dim() == 2 then
      self.output:index(self.weight, 1, input:view(-1))
      self.output = self.output:view(input:size(1), input:size(2), self.weight:size(2))
   else
      error("input must be a vector or matrix")
   end
   return self.output
end

function PreLookupTable:updateGradInput(input, gradOutput)
   -- the input can be of any type (as in the forward it's 
   -- converted anyway to LongTensor) thus, need to allocate
   -- new memory each time the user changes the input type
   if torch.type(self.gradInput) ~= torch.type(input) then
      self.gradInput = input.new()
   end
   if not self.gradInput:isSameSizeAs(input) then
      self.gradInput:resizeAs(input):zero()
   end
   return self.gradInput
end

function PreLookupTable:accGradParameters(input, gradOutput, scale)
   self:backCompatibility()
   input = self.copiedInput and self._input or input
   if input:dim() == 2 then
      input = input:view(-1)
   elseif input:dim() ~= 1 then
      error("input must be a vector or matrix")
   end

   --if not gradOutput:isContiguous() then
   --   self._gradOutput = self._gradOutput or gradOutput.new()
   --   self._gradOutput:resizeAs(gradOutput):copy(gradOutput)
   --   gradOutput = self._gradOutput
   --end

   --self.gradWeight.THNN.LookupTable_accGradParameters(
   --   input:cdata(),
   --   gradOutput:cdata(),
   --   self.gradWeight:cdata(),
   --   self._count:cdata(),
   --   THNN.optionalTensor(self._sorted),
   --   THNN.optionalTensor(self._indices),
   --   self.shouldScaleGradByFreq or false,
   --   self.paddingValue or 0,
   --   scale or 1
   --)
end

function PreLookupTable:renorm(input)
   if not self.maxNorm then
      return
   end
   -- copy input into _input, so _input is continous.
   -- The copied _input will be modified in the C code.
   self._input:resize(input:size()):copy(input)
   local row_idx = self._input
   if row_idx:dim() == 2 then
      row_idx = row_idx:view(-1)
   elseif row_idx:dim() ~= 1 then
      error("input must be a vector or matrix")
   end
   -- "row_idx" and "weight" will be modified in the C code
   self.weight.THNN.PreLookupTable_renorm(
      row_idx:cdata(),
      self.weight:cdata(),
      self.maxNorm,
      self.normType or 2
   )
end

function PreLookupTable:type(type, tensorCache)
   parent.type(self, type, tensorCache)

   if type == 'torch.CudaTensor' then
      -- CUDA uses _sorted and _indices temporary tensors
      self._sorted = self.weight.new()
      self._indices = self.weight.new()
      self._count = self.weight.new()
      self._input = self.weight.new()
   else
      -- self._count and self._input should only be converted if using Cuda
      self._count = torch.IntTensor()
      self._input = torch.LongTensor()
   end

   return self
end

function PreLookupTable:clearState()
   nn.utils.clear(self, '_count', '_input', '_gradOutput')
   return parent.clearState(self)
end

-- we do not need to accumulate parameters when sharing
PreLookupTable.sharedAccUpdateGradParameters = PreLookupTable.accUpdateGradParameters