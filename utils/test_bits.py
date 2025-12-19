from utils.bits import string_to_bits, bits_to_string

L = 128
s = "HELLO WORLD! Привет!"

bits = string_to_bits(s, L)
s2 = bits_to_string(bits)

print("Original:", s)
print("Decoded :", s2)
print("Bits shape:", bits.shape)
