IntervalDim = 100
NoteOnDim = 128
NoteOffDim = 128

NoteOnOffset = IntervalDim
NoteOffOffset = IntervalDim + NoteOnOffset

# type = 'poetry_strains'
# type = 'poetry_paras'
# type = 'random_poetry'
# type = 'random_music'
# mode = "pretrain"

type = 'music'
mode = "finetune"

maxLength = Time = 100
EmbeddingDim = 512 
HeadDim = 32 
Heads = 16 
ContextDim = HeadDim * Heads # 512
Layers = 6 # how many decoders

batch_size = batchSize = 64