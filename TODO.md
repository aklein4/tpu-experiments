
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
 - [x] Elementwise attention bias
 - [x] Implement LLM loss
    - [x] Mask loss with padding
    - [x] log acc
    - [x] log pcorr
 - [ ] Train ZLM-equivalent Llama model

### ZLM
 - [x] Implement ZLM Model
    - [x] LoRa QKV and up/gate
    - [x] Account for pad tokens in position_ids
    - [x] attention bias for pad tokens
    - [x] Implement Decoder
    - [x] Implement Generator
    - [x] Implement Encoder
    - [x] Implement parameterized alpha for RMS scaling
 - [ ] Implement ZLM Loss
    - [ ] LM loss, acc, pcorr
    - [ ] KL loss hook for alpha?
    - [ ] weight of grad(KL, gen_mu) = 1
    - [ ] weight of grad(KL, alpha) = scale
    - [ ] weight of grad(KL, enc_mu) = [0, warmup, scale] with batch L1 weighting
    - [ ] acc threshold for grad(LM loss, enc_mu)? (save grad scale in previous wtep closure)
 - [ ] Train ZLM

### Issues
 - [ ] Why are there NaNs?