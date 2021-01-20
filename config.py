IntervalDim = 100
VelocityDim = 32
NoteOnDim = 128
NoteOffDim = 128

VelocityOffset = IntervalDim
NoteOnOffset = IntervalDim + VelocityDim
NoteOffOffset = IntervalDim + VelocityDim + NoteOnDim
EventDim = IntervalDim + VelocityDim + NoteOnDim + NoteOffDim # 388

maxLength = Time = 100 #处理的每个sample的维度
EmbeddingDim = 512 #把每个sample嵌入的维度
HeadDim = 32 #头的维度
Heads = 16 #多少个头
ContextDim = HeadDim * Heads # 512
Layers = 8 #多少个decoder

batch_size = batchSize = 1