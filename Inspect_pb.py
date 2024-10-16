import tensorflow as tf

def inspect_saved_model(saved_model_dir):
    # Load the SavedModel
    model = tf.saved_model.load(saved_model_dir)
    
    print("Loaded SavedModel")
    print("Concrete Functions:")
    for signature_key in model.signatures:
        print(f"Signature key: {signature_key}")
        concrete_function = model.signatures[signature_key]
        print("  Inputs:")
        for input_key, input_tensor in concrete_function.inputs.items():
            print(f"    {input_key}: {input_tensor.shape} {input_tensor.dtype}")
        print("  Outputs:")
        for output_key, output_tensor in concrete_function.outputs.items():
            print(f"    {output_key}: {output_tensor.shape} {output_tensor.dtype}")
        print()

    print("Model structure:")
    for layer in model.variables:
        print(f"  {layer.name}: {layer.shape}")

# Usage
inspect_saved_model(model.pb)
