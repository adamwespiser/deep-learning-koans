# # Tutorial X: Header
#
# In this tutorial we will cover the following
#  - *What are Recurrent Neural Networks?* What is the fundamental problem they solve? LSTM, RNN building blocks, (aside on LTSM white papers, advanced methods like attention, word vecs, doc vecs, et cetera). Communication of RNN vs CNN differences
#  - Flux RNN Layer Interface within Flux: [RNN](https://github.com/FluxML/Flux.jl/blob/master/src/layers/recurrent.jl#L70) [LSTM](https://github.com/FluxML/Flux.jl/blob/master/src/layers/recurrent.jl#L109) [GRU](https://github.com/FluxML/Flux.jl/blob/master/src/layers/recurrent.jl#L156)
#  - Put it all together: [Char-Rnn, from model-zoo](https://github.com/FluxML/model-zoo/blob/master/text/char-rnn/char-rnn.jl)
#  - Mention: Loss functions, training functions, sampling, character/word encoding
#
# Problem Statement: < fill this in >
#
# # Setup
# <load all the libraries here>

using Flux

# # First Example
# for the next examples
x = 1
print(x)
