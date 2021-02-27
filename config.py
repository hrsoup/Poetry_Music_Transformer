exp_number = 1 

IntervalDim = 100
NoteOnDim = 128
NoteOffDim = 128

NoteOnOffset = IntervalDim
NoteOffOffset = IntervalDim + NoteOnOffset

maxLength = Time = 100
EmbeddingDim = 512 
Heads = 16 
Layers = 6 # how many decoders

batch_size = batchSize = 64