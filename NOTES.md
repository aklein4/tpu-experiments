
### Startup

1. Set up logging (not important)

2. Validate Config
 - just checks context parallelism

3. Print config

4. More logging setup

5. Set random seed

6. Set the default dtype to float32
 - TODO: why?

7. Initialize model
 - wraps initialization (and therefore model weights) in torch_dtype -> model_dtype 
 - wraps initialization in torch_xla.device()
   - TODO: what does this do and why?
 - rendezvous devices
   - TODO: how does this compare to other sync methods?
 - log model info

8. Initialize trainer
 - apply_xla_patch_to_nn_linear
   - to avoid breaking sharding
 - auto_trace
   - just for profiling/debugging?
 -  