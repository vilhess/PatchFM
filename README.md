# RESULTS ON ARTIFICIAL DATASET
PatchFM : 200 total epochs 4 warmups steps for 80 epochs batch size 256

MSE Loss

# context length 1024 forecast horizon 32
PatchFM : 86787.2265625
TiRex : 1500157.37

# context length 512 forecast horizon 32
PatchFM : 197384.328125
TiRex : 2596248.5

# context length 256 forecast horizon 32
PatchFM : 210300.03125
TiRex : 2552907.75

# context length 128 forecast horizon 32
PatchFM : 151495.59375
TiRex : 3327954.25


# artificial model on utsd, ctx len 1024 : 205519.0625
# all model on utsd, ctx len 1024 : 143534.796875
# tirex on utsd, ctx len 1024 : 106892.84375
# big model on utsd, ctx len 1024 : 140541.9375