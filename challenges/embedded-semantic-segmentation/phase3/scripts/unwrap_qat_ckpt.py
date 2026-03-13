import torch
import argparse
import os

def unwrap_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if '.mod.' in k:
            new_key = k.replace('.mod.', '.')
            new_state_dict[new_key] = v
        elif 'fake_quant' in k or 'observer' in k:
            continue
        else:
            new_state_dict[k] = v
    return new_state_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Input QAT checkpoint')
    parser.add_argument('output', help='Output unwrapped checkpoint')
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location='cpu')
    ckpt['state_dict'] = unwrap_state_dict(ckpt['state_dict'])
    
    # Also fix meta if needed
    if 'meta' in ckpt:
        # Check if we need to adjust something
        pass

    torch.save(ckpt, args.output)
    print(f"Unwrapped checkpoint saved to: {args.output}")

if __name__ == '__main__':
    main()
