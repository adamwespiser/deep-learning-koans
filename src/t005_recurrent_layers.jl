# # Tutorial X: Header
#
# In this tutorial we will cover the following
#  - *What are Recurrent Neural Networks?* What is the fundamental problem they solve? LSTM, RNN building blocks, (aside on LTSM white papers, advanced methods like attention, word vecs, doc vecs, et cetera). Communication of RNN vs CNN differences
#  - Flux RNN Layer Interface within Flux: [RNN](https://github.com/FluxML/Flux.jl/blob/master/src/layers/recurrent.jl#L70) [LSTM](https://github.com/FluxML/Flux.jl/blob/master/src/layers/recurrent.jl#L109) [GRU](https://github.com/FluxML/Flux.jl/blob/master/src/layers/recurrent.jl#L156)
#  - Put it all together: [Char-Rnn, from model-zoo](https://github.com/FluxML/model-zoo/blob/master/text/char-rnn/char-rnn.jl)
#  - Mention: Loss functions, training functions, sampling, character/word encoding
#
# Problem Statement: To Understand Flux's Recurraent NN layers
# [Backround Reading](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
# [Flux Documentation](https://fluxml.ai/Flux.jl/v0.4/models/layers.html#Recurrent-Layers-1)
# # Setup
# Import the libraries here, from `Flux`, `StatsBase`, and `Base.Iterators`
using Flux
using Flux: onehot, chunk, batchseq, throttle, crossentropy
using StatsBase: wsample
using Base.Iterators: partition
text_file_local = "../src/assets/t005_recurrent_layers/shakespeare.txt"
text_file_remote_url = "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt"

# # Download shakespeare dataset
# Run this once, per time you download the project
isfile(text_file_local) ||
  download(text_file_remote_url,
          text_file_local)


# # Set up Shakespeare dataset
text = collect(String(read(text_file_local)))
alphabet = [unique(text)..., '_']
text = map(ch -> onehot(ch, alphabet), text)
stop = onehot('_', alphabet)

N = length(alphabet)
seqlen = 50
nbatch = 50

Xs = collect(partition(batchseq(chunk(text,        nbatch), stop), seqlen))
Ys = collect(partition(batchseq(chunk(text[2:end], nbatch), stop), seqlen))

# Note, the following types match, for the encoding for `Xs`, a character, and
# `Ys`, the subsequent character...
# typeof(text)                                     :: Array{Flux.OneHotVector,1}
# typeof(chunk(text,nbatch))                       :: Array{Array{Flux.OneHotVector,1},1}
# typeof(batchseq(chunk(text,nbatch), stop))       :: Array{Flux.OneHotMatrix{Array{Flux.OneHotVector,1}},1}
# typeof(collect(partition(batchseq(chunk(text,nbatch), stop), seqlen))) :: Array{Array{Flux.OneHotMatrix{Array{Flux.OneHotVector,1}},1},1}

# # One-Hot Encoding.
# We read our shakespeare file into a text string, and apply One-Hot text encoding on it.
# Using the variables, text, and `alphabet`, the one-hot encoding scheme, reverse the
# get a String of the characters between 100 and 117
idx_map = collect(100:118) # collect makes an Array from a UnitRange
string_msg = String(['a', 'b']) # Modify me !
string_msg = String(map(x -> alphabet[x][1], text[idx_map])) #src
@assert string_msg == "u are all resolved"

# # Working with OneHot Vectors
# Get the first letter of the dataset, from Xs
xs_letter = "Not a letter" # Fix Me  !!
xs_letter = alphabet[Xs[1][1][:,1]]
@assert alphabet[text[1]] == xs_letter


# # Flux Recurrances
# Flux creates recurrant layers using the constructor, `Recur`
# In the following example, we will create a simple recurrant
# cell, for addition, what will be the result of applying it to
# the sequence of numbers from 1:10 ?
cell(h, x) = (h + x, x)
rnn = Flux.Recur(cell, 0)
rnn(collect(1:100))

rnn_state = 0 # Modify me !
rnn_state = 5050 #src
@assert rnn.state = rnn_state


# # Flux Recurance, many dimensions
# Given a `n` dimensional input, create a `Flux.Recur` cal
# modifying the code above
n = 10
seq_len = 20
input_data = ones(n, seq_len)

cell(h, x) = (h .+ x, x)
rnn = Flux.Recur(cell, zeros(n))

rnn.(input_data) # modify the shape of the input data
rnn.([input_data[:,i] for i in 1:10]) #src
@assert ones(seq_len)*10 == rnn.state
# Note, the desired input type passed to rnn is going to be `Array{Array{Float64,1},1}``

# # RNN - Basic layers
# apply the layer to Xs[1]
layer = RNN(N, 1)
layer.(Xs[1])




# # Adding Layers to Models
# Given a model, `m`, a loss function

m = Chain(
  LSTM(N, 128),
  LSTM(128, 128),
  Dense(128, N),
  softmax)


function loss(xs, ys)
  l = sum(crossentropy.(m.(xs), ys))
  Flux.truncate!(m)
  return l
end

opt = ADAM(0.01)
tx, ty = (Xs[5], Ys[5])
evalcb = () -> @show loss(tx, ty)

params_m = params(m)

Flux.train!(loss = NaN, params = NaN, data = NaN, opts = NaN, cb = throttle(evalcb, 30)) # Fill in the first 4 args
Flux.train!(loss, params(m), zip(Xs, Ys), opt, cb = throttle(evalcb, 30)) #src

@assert params_m != params(m)

