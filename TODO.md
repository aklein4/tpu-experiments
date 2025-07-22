
### Repo Setup
 - [x] Port torchprime
 - [x] Implement data loading/collating
    - [x] non-streaming (small dataset)
    - [x] streaming (large dataset)
 - [x] Add wandb logging
 - [x] Check inter-device syncing
 - [ ] Implement checkpoint saving
    - [x] Basic saving
    - [ ] Convert to safetensors
 - [ ] Implement checkpoint loading

### LLM
 - [x] Modify Llama to have combined qkv/gate_up for later mods
 - [x] Mask in attention (with segment ids)
 - [ ] Implement LLM loss
    - [x] Mask loss with padding
    - [ ] log acc
    - [ ] log pcorr
 - [ ] Train ZLM-equivalent Llama model

### ZLM
 - [ ] Implement ZLM Model
    - [ ] LoRa QKV and up/gate
    - [ ] Account for pad in position_ids
    - [ ] segment_ids from pad
    - [ ] Implement Decoder
    - [ ] Implement Generator
    - [ ] Implement Encoder
    - [ ] Implement parameterized alpha for RMS scaling
 - [ ] Implement ZLM Loss
    - [ ] LM loss, acc, pcorr
    - [ ] KL loss hook for alpha?
    - [ ] weight of grad(KL, gen_mu) = 1
    - [ ] weight of grad(KL, alpha) = scale
    - [ ] weight of grad(KL, enc_mu) = [0, warmup, scale] with batch L1 weighting
    - [ ] acc threshold for grad(LM loss, enc_mu)? (save grad scale in previous wtep closure)
 - [ ] Train ZLM

