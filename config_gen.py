
CONFIG = """
'*.layers.*.self_attn.o_proj.weight': [fsdp, null]
'*.layers.*.mlp.down_proj.weight': [null, fsdp]

'*.layers.*.input_layernorm.weight': [fsdp]
'*.layers.*.post_attention_layernorm.weight': [fsdp]
'*.norm.weight': [fsdp]

'*.layers.*.self_attn.qkv_proj.base_linear.weight': [fsdp, null]
'*.layers.*.self_attn.qkv_proj.lora_down.weight': [null, fsdp]
'*.layers.*.self_attn.qkv_proj.lora_up.weight': [fsdp, null]

'*.layers.*.mlp.gate_up_proj.base_linear.weight': [fsdp, null]
'*.layers.*.mlp.gate_up_proj.lora_down.weight': [null, fsdp]
'*.layers.*.mlp.gate_up_proj.lora_up.weight': [fsdp, null]
"""


def main():
    config = CONFIG.replace("'", '')

    out = ""

    for t in ['encoder', 'generator', 'decoder']:
        
        lines = config.strip().split('\n')
        for line in lines:
            if line.strip():
                modified_line = line.replace('*', t, 1)
                out += modified_line + '\n'
            else:
                out += '\n'

    with open('config_out.txt', 'w') as f:
        f.write(out.strip())


if __name__ == "__main__":
    main()