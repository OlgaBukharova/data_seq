from utils.payload import make_random_string_batch

L = 256
bits, texts = make_random_string_batch(batch_size=4, n_chars=16, L=L)

print(texts)
print(bits.shape, bits.dtype, bits.min().item(), bits.max().item())
