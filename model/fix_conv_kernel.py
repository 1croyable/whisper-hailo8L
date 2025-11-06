import onnx
from onnx import helper
from onnx import numpy_helper

def main():
    src = 'whisper-hailo8l-trained0.onnx'
    dst = 'whisper-hailo8l-trained0.onnx'
    print(f'Loading {src} (with external data if present)...')
    m = onnx.load(src, load_external_data=True)

    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}

    fixed_count = 0
    cannot_infer = []

    for node in m.graph.node:
        if node.op_type in ('Conv', 'ConvTranspose'):
            has_ks = any(a.name == 'kernel_shape' for a in node.attribute)
            if has_ks:
                continue
            weight_name = node.input[1] if len(node.input) > 1 else None
            if weight_name and weight_name in inits:
                wshape = inits[weight_name].shape
                if len(wshape) >= 3:
                    kernel_shape = list(wshape[2:])
                    node.attribute.append(helper.make_attribute('kernel_shape', kernel_shape))
                    fixed_count += 1
                    print(f'Fixed node {node.name!r} op={node.op_type} kernel_shape={kernel_shape}')
                else:
                    cannot_infer.append((node.name, node.op_type, wshape))
            else:
                cannot_infer.append((node.name, node.op_type, weight_name))

    print(f'Applied fixes to {fixed_count} nodes.')
    if cannot_infer:
        print('Could not infer kernel_shape for the following nodes (weight missing or invalid):')
        for info in cannot_infer:
            print(' ', info)

    print(f'Saving fixed model to {dst}...')
    onnx.save(m, dst)
    print('Saved.')

if __name__ == '__main__':
    main()
