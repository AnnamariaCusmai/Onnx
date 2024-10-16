import tensorflow as tf

def inspect_saved_model(saved_model_dir):
    model = tf.saved_model.load(saved_model_dir)
    
    print("Operations in the loaded model:")
    for concrete_func in model.signatures.values():
        print(concrete_func.structured_outputs)
    
    print("\nTensors in the graph:")
    for func in model.signatures.values():
        for input_tensor in func.inputs:
            print(f"Input Tensor: {input_tensor}")
        for output_tensor in func.outputs:
            print(f"Output Tensor: {output_tensor}")

# Usage
inspect_saved_model("model.pb")
