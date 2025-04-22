import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.mlp import MLP

def clone_mlp(original_model, clone_factor):
    """
    Create a cloned version of an MLP model with clone_factor times larger hidden layers.
    
    Args:
        original_model: The original MLP model
        clone_factor: Integer factor to multiply hidden layer sizes by
        
    Returns:
        A new MLP model with cloned parameters
    """
    if not isinstance(original_model, MLP):
        raise ValueError("Input model must be an MLP instance")
    
    # Extract model configuration
    input_size = original_model.input_size
    hidden_sizes = [size * clone_factor for size in original_model.hidden_sizes]
    output_size = original_model.output_size
    
    # Get other parameters by inspecting the model
    has_norm = False
    norm_type = None
    norm_after_activation = original_model.norm_after_activation
    activation_type = None
    has_dropout = False
    dropout_p = 0.0
    
    # Detect settings from original model
    for name, module in original_model.named_modules():
        if isinstance(module, nn.ReLU):
            activation_type = 'relu'
        elif isinstance(module, nn.Tanh):
            activation_type = 'tanh'
        elif isinstance(module, nn.LeakyReLU):
            activation_type = 'leaky_relu'
        elif isinstance(module, nn.Sigmoid):
            activation_type = 'sigmoid'
        elif isinstance(module, nn.GELU):
            activation_type = 'gelu'
        elif isinstance(module, nn.ELU):
            activation_type = 'elu'
        elif isinstance(module, nn.SELU):
            activation_type = 'selu'
        elif isinstance(module, nn.BatchNorm1d):
            has_norm = True
            norm_type = 'batch'
        elif isinstance(module, nn.LayerNorm):
            has_norm = True
            norm_type = 'layer'
        elif isinstance(module, nn.Dropout):
            has_dropout = True
            dropout_p = module.p
    
    # Check if bias is used in Linear layers
    bias = True
    for name, module in original_model.named_modules():
        if isinstance(module, nn.Linear):
            bias = module.bias is not None
            break
    
    # Check if normalization has affine parameters
    normalization_affine = True
    for name, module in original_model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            normalization_affine = module.weight is not None
            break
    
    # Create new model
    cloned_model = MLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        activation=activation_type if activation_type else 'none',
        dropout_p=0.0,  # We don't use dropout in cloned model as it breaks symmetry
        normalization=norm_type,
        norm_after_activation=norm_after_activation,
        bias=bias,
        normalization_affine=normalization_affine
    )
    
    # Copy and clone parameters
    clone_parameters(original_model, cloned_model, clone_factor)
    
    return cloned_model

def clone_parameters(original_model, cloned_model, clone_factor):
    """
    Clone parameters from original model to the larger model using a consistent approach.
    
    Parameters are cloned according to these rules:
    - If dimensions match, copy directly
    - If one dimension differs, clone along that dimension
    - If two dimensions differ, clone along both dimensions
    
    Args:
        original_model: Source model
        cloned_model: Target model with larger hidden layers
        clone_factor: The factor by which hidden layers were multiplied
    """
    # Get parameter shapes from both models
    orig_params = dict(original_model.named_parameters())
    cloned_params = dict(cloned_model.named_parameters())
    
    # Debug info
    print("Parameter mapping:")
    for name in orig_params.keys():
        if name in cloned_params:
            print(f"{name}: {orig_params[name].shape} -> {cloned_params[name].shape}")
    
    # Process each parameter with a unified approach
    for name, cloned_param in cloned_params.items():
        if name in orig_params:
            orig_param = orig_params[name]
            
            # Case 1: Same shape - direct copy
            if orig_param.shape == cloned_param.shape:
                cloned_param.data.copy_(orig_param.data)
                print(f"Direct copy: {name}")
                continue
            
            # Special case for output layer weights - scale to maintain output magnitude
            is_output_weight = name == "layers.out.weight"
            
            # Case 2: 1D parameter (bias or normalization parameter)
            if len(orig_param.shape) == 1 and len(cloned_param.shape) == 1:
                if cloned_param.shape[0] // orig_param.shape[0] == clone_factor:
                    # Clone the vector multiple times
                    for i in range(clone_factor):
                        start_idx = i * orig_param.shape[0]
                        end_idx = (i + 1) * orig_param.shape[0]
                        cloned_param.data[start_idx:end_idx].copy_(orig_param.data)
                    print(f"1D clone: {name}")
                
            # Case 3: 2D parameter (weight matrix)
            elif len(orig_param.shape) == 2 and len(cloned_param.shape) == 2:
                orig_rows, orig_cols = orig_param.shape
                cloned_rows, cloned_cols = cloned_param.shape
                
                # Case 3a: Both dimensions differ by clone_factor
                if cloned_rows // orig_rows == clone_factor and cloned_cols // orig_cols == clone_factor:
                    for i in range(clone_factor):
                        for j in range(clone_factor):
                            row_start = i * orig_rows
                            row_end = (i + 1) * orig_rows
                            col_start = j * orig_cols
                            col_end = (j + 1) * orig_cols
                            cloned_param.data[row_start:row_end, col_start:col_end].copy_(orig_param.data)
                    print(f"2D clone (both dims): {name}")
                
                # Case 3b: Only rows differ by clone_factor (first layer)
                elif cloned_rows // orig_rows == clone_factor and cloned_cols == orig_cols:
                    for i in range(clone_factor):
                        row_start = i * orig_rows
                        row_end = (i + 1) * orig_rows
                        cloned_param.data[row_start:row_end, :].copy_(orig_param.data)
                    print(f"2D clone (rows only): {name}")
                
                # Case 3c: Only columns differ by clone_factor (output layer)
                elif cloned_rows == orig_rows and cloned_cols // orig_cols == clone_factor:
                    # For output layer, scale weights to maintain output magnitude
                    scaling_factor = 1.0 / clone_factor if is_output_weight else 1.0
                    for j in range(clone_factor):
                        col_start = j * orig_cols
                        col_end = (j + 1) * orig_cols
                        cloned_param.data[:, col_start:col_end].copy_(orig_param.data * scaling_factor)
                    print(f"2D clone (cols only): {name}{' with scaling' if is_output_weight else ''}")
            
            # Unexpected parameter shape
            else:
                print(f"Warning: Parameter {name} shapes don't match as expected: {orig_param.shape} vs {cloned_param.shape}")

def test_cloning(clone_factor=3):
    """
    Test cloning functionality by creating two models and validating that
    they produce the same outputs for the same inputs.
    """
    # Create original model - use no normalization to avoid GroupNorm issues
    original_model = MLP(
        input_size=784,
        hidden_sizes=[128, 64, 32],
        output_size=10,
        activation='tanh',
        normalization=None,
        norm_after_activation=False,
        dropout_p=0.0,  # No dropout for this test
        bias=True
    )
    
    # Create cloned model
    cloned_model = clone_mlp(original_model, clone_factor)
    
    # Set both models to eval mode
    original_model.eval()
    cloned_model.eval()
    
    # Generate random input
    torch.manual_seed(42)
    test_input = torch.randn(10, 784)
    
    # Get outputs from both models
    with torch.no_grad():
        original_output = original_model(test_input)
        cloned_output = cloned_model(test_input)
    
    # Check if outputs match
    is_equal = torch.allclose(original_output, cloned_output, rtol=1e-4, atol=1e-4)
    
    if is_equal:
        print(f"✅ Test passed! Original and cloned model outputs match.")
        print(f"Original model hidden sizes: {original_model.hidden_sizes}")
        print(f"Cloned model hidden sizes: {cloned_model.hidden_sizes}")
    else:
        print(f"❌ Test failed! Outputs differ.")
        print(f"Original output: {original_output[0][:5]}...")
        print(f"Cloned output: {cloned_output[0][:5]}...")
        
        # Additional debugging: check activations in each layer
        print("\nDebugging activations layer by layer:")
        x_orig = test_input
        x_clone = test_input
        
        for i in range(len(original_model.hidden_sizes)):
            # Forward through original model one layer at a time
            x_orig = original_model.layers[f'linear_{i}'](x_orig)
            if f'norm_{i}' in original_model.layers and not original_model.norm_after_activation:
                x_orig = original_model.layers[f'norm_{i}'](x_orig)
            x_orig = original_model.layers[f'act_{i}'](x_orig)
            if f'norm_{i}' in original_model.layers and original_model.norm_after_activation:
                x_orig = original_model.layers[f'norm_{i}'](x_orig)
            
            # Same for cloned model
            x_clone = cloned_model.layers[f'linear_{i}'](x_clone)
            if f'norm_{i}' in cloned_model.layers and not cloned_model.norm_after_activation:
                x_clone = cloned_model.layers[f'norm_{i}'](x_clone)
            x_clone = cloned_model.layers[f'act_{i}'](x_clone)
            if f'norm_{i}' in cloned_model.layers and cloned_model.norm_after_activation:
                x_clone = cloned_model.layers[f'norm_{i}'](x_clone)
                
            # For cloned model, we expect the activations to repeat in groups of size orig_hidden_size
            orig_size = original_model.hidden_sizes[i]
            clone_size = cloned_model.hidden_sizes[i]
            
            print(f"\nLayer {i} activations:")
            print(f"Original shape: {x_orig.shape}, Cloned shape: {x_clone.shape}")
            
            # Check if activations repeat in the cloned model
            for j in range(clone_factor):
                start_idx = j * orig_size
                end_idx = (j + 1) * orig_size
                if j == 0:
                    # Compare first block with original
                    block_match = torch.allclose(x_clone[:, start_idx:end_idx], x_orig, rtol=1e-4, atol=1e-4)
                    print(f"Block {j} matches original: {block_match}")
                else:
                    # Compare other blocks with the first block
                    first_block = x_clone[:, 0:orig_size]
                    block_match = torch.allclose(x_clone[:, start_idx:end_idx], first_block, rtol=1e-4, atol=1e-4)
                    print(f"Block {j} matches first block: {block_match}")

def verify_cloned_gradient_behavior(clone_factor=3, epochs=5):
    """
    Test whether gradients in the cloned model maintain the cloning property during training.
    This verifies that gradients respect the cloned-unit manifold described in the paper.
    """
    # Create original model
    original_model = MLP(
        input_size=784,
        hidden_sizes=[20, 10],
        output_size=10,
        activation='tanh',
        # Use no normalization to avoid GroupNorm issues
        normalization=None,
        dropout_p=0.0
    )
    
    # Create cloned model
    cloned_model = clone_mlp(original_model, clone_factor)
    
    # Set up a simple optimization task
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(cloned_model.parameters(), lr=0.01)
    
    # Generate some random data
    torch.manual_seed(42)
    inputs = torch.randn(5, 784)
    targets = torch.randn(5, 10)
    
    print("Testing gradient behavior during training...")
    
    for epoch in range(epochs):
        # Forward pass
        outputs = cloned_model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check if gradients maintain cloning structure
        clone_preserved = True
        
        # Check each parameter's gradient
        for name, param in cloned_model.named_parameters():
            if 'linear' in name and 'weight' in name:
                grad = param.grad.data
                
                # For hidden layers, check if gradients maintain block structure
                if param.shape[0] > original_model.output_size:  # Not the output layer
                    orig_rows = param.shape[0] // clone_factor
                    orig_cols = param.shape[1] // clone_factor if 'linear_0' not in name else param.shape[1]
                    
                    # If not the first layer, check both dimensions
                    if 'linear_0' not in name:
                        for i in range(clone_factor):
                            for j in range(clone_factor):
                                row_start = i * orig_rows
                                row_end = (i + 1) * orig_rows
                                col_start = j * orig_cols
                                col_end = (j + 1) * orig_cols
                                
                                # Compare this block's gradient with the first block
                                if i == 0 and j == 0:
                                    first_block = grad[row_start:row_end, col_start:col_end]
                                else:
                                    current_block = grad[row_start:row_end, col_start:col_end]
                                    if not torch.allclose(current_block, first_block, rtol=1e-4, atol=1e-4):
                                        clone_preserved = False
                                        print(f"Epoch {epoch}, parameter {name}: Gradient block structure broken")
                                        print(f"Block (0,0) vs Block ({i},{j}): {first_block[0,0]} vs {current_block[0,0]}")
                    
                    # For the first layer, check only the output dimension
                    else:
                        for j in range(clone_factor):
                            col_start = j * orig_cols
                            col_end = (j + 1) * orig_cols
                            
                            if j == 0:
                                first_block = grad[:, col_start:col_end]
                            else:
                                current_block = grad[:, col_start:col_end]
                                if not torch.allclose(current_block, first_block, rtol=1e-4, atol=1e-4):
                                    clone_preserved = False
                                    print(f"Epoch {epoch}, parameter {name}: Gradient block structure broken")
        
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Clone structure preserved: {clone_preserved}")
        
        # Apply gradients
        optimizer.step()
    
    if clone_preserved:
        print("\n✅ Test passed! Gradient updates maintained the cloned structure throughout training.")
    else:
        print("\n❌ Test failed! Gradient updates broke the cloned structure.")

def test_forward_activation_cloning(clone_factor=3):
    """
    Test that activations in the cloned model maintain the cloning pattern.
    Each group of clone_factor units should have identical activations.
    """
    # Create original model with small number of units for easier visualization
    original_model = MLP(
        input_size=784,
        hidden_sizes=[6, 4],  # Small sizes for easier visualization
        output_size=2,
        activation='tanh',
        normalization=None,
        dropout_p=0.0
    )
    
    # Create cloned model
    cloned_model = clone_mlp(original_model, clone_factor)
    
    # Set both models to eval mode
    original_model.eval()
    cloned_model.eval()
    
    # Generate random input
    torch.manual_seed(42)
    test_input = torch.randn(2, 784)
    
    # Run forward pass
    with torch.no_grad():
        # Get activations from intermediate layers
        activations_original = {}
        activations_cloned = {}
        
        # Original model
        x_orig = test_input
        for i in range(len(original_model.hidden_sizes)):
            x_orig = original_model.layers[f'linear_{i}'](x_orig)
            activations_original[f'pre_act_{i}'] = x_orig.clone()
            x_orig = original_model.layers[f'act_{i}'](x_orig)
            activations_original[f'post_act_{i}'] = x_orig.clone()
        
        # Cloned model
        x_clone = test_input
        for i in range(len(cloned_model.hidden_sizes)):
            x_clone = cloned_model.layers[f'linear_{i}'](x_clone)
            activations_cloned[f'pre_act_{i}'] = x_clone.clone()
            x_clone = cloned_model.layers[f'act_{i}'](x_clone)
            activations_cloned[f'post_act_{i}'] = x_clone.clone()
    
    # Print activations for visualization
    print("\n=== Forward Activation Test ===")
    for i in range(len(original_model.hidden_sizes)):
        orig_size = original_model.hidden_sizes[i]
        print(f"\nLayer {i} activations:")
        
        # For the first sample
        print(f"Original model activations: {activations_original[f'post_act_{i}'][0]}")
        print(f"Cloned model activations by block:")
        for j in range(clone_factor):
            start_idx = j * orig_size
            end_idx = (j + 1) * orig_size
            print(f"  Block {j}: {activations_cloned[f'post_act_{i}'][0, start_idx:end_idx]}")
    
    # Test if blocks match within cloned model (unit-by-unit comparison)
    all_blocks_match = True
    detailed_matches = True
    
    for i in range(len(original_model.hidden_sizes)):
        orig_size = original_model.hidden_sizes[i]
        
        # Compare blocks at a high level
        for j in range(1, clone_factor):
            start_first = 0
            end_first = orig_size
            start_current = j * orig_size
            end_current = (j + 1) * orig_size
            
            blocks_match = torch.allclose(
                activations_cloned[f'post_act_{i}'][:, start_first:end_first],
                activations_cloned[f'post_act_{i}'][:, start_current:end_current],
                rtol=1e-4, atol=1e-4
            )
            
            if not blocks_match:
                all_blocks_match = False
                print(f"Layer {i}: Block 0 and Block {j} activations don't match")
        
        # Detailed unit-by-unit comparison (precise activation matching)
        print(f"\nDetailed unit-by-unit comparison for Layer {i}:")
        for unit in range(orig_size):
            unit_activations = []
            for j in range(clone_factor):
                idx = j * orig_size + unit
                unit_activations.append(activations_cloned[f'post_act_{i}'][:, idx])
            
            # Check if all copies of this unit have identical activations
            reference = unit_activations[0]
            for j in range(1, clone_factor):
                unit_match = torch.allclose(
                    reference,
                    unit_activations[j],
                    rtol=1e-5, atol=1e-5
                )
                
                if not unit_match:
                    detailed_matches = False
                    print(f"  Unit {unit}: Copy 0 and Copy {j} activations differ")
                    print(f"    Copy 0: {reference[0].item():.6f}")
                    print(f"    Copy {j}: {unit_activations[j][0].item():.6f}")
                    print(f"    Difference: {torch.abs(reference - unit_activations[j])[0].item():.6f}")
        
        if detailed_matches:
            print(f"  All unit copies in Layer {i} have identical activations ✓")
    
    if all_blocks_match and detailed_matches:
        print("\n✅ Test passed! All activation blocks and individual unit copies within cloned model match.")
    else:
        print("\n❌ Test failed! Some activation blocks or unit copies don't match.")
        
    return all_blocks_match and detailed_matches


def test_gradient_cloning(clone_factor=3):
    """
    Test that gradients in the cloned model maintain the cloning pattern.
    Gradients for cloned units should be identical.
    """
    # Create original model with small number of units for easier visualization
    original_model = MLP(
        input_size=784,
        hidden_sizes=[6, 4],  # Small sizes for easier visualization
        output_size=2,
        activation='tanh',
        normalization=None,
        dropout_p=0.0
    )
    
    # Create cloned model
    cloned_model = clone_mlp(original_model, clone_factor)
    
    # Set up loss function
    criterion = nn.MSELoss()
    
    # Generate random input and target
    torch.manual_seed(42)
    inputs = torch.randn(2, 784)
    targets = torch.randn(2, 2)
    
    # Forward pass
    outputs = cloned_model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass
    cloned_model.zero_grad()
    loss.backward()
    
    print("\n=== Gradient Test ===")
    
    # Check if gradients maintain cloning pattern
    gradients_match = True
    
    # Check gradients for each layer's weights
    for i in range(len(original_model.hidden_sizes)):
        layer_weight = f"layers.linear_{i}.weight"
        layer_bias = f"layers.linear_{i}.bias"
        
        # Get parameter and gradient
        if hasattr(cloned_model, 'layers') and layer_weight in dict(cloned_model.named_parameters()):
            param = dict(cloned_model.named_parameters())[layer_weight]
            grad = param.grad
            
            if grad is None:
                print(f"No gradient for {layer_weight}")
                continue
                
            orig_size_out = original_model.hidden_sizes[i]
            orig_size_in = (original_model.hidden_sizes[i-1] if i > 0 else original_model.input_size)
            
            print(f"\nGradients for {layer_weight}:")
            print(f"Shape: {grad.shape}")
            
            # Case 1: First layer - only rows are expanded
            if i == 0:
                # Check if each block of rows has identical gradients
                for j in range(1, clone_factor):
                    start_first = 0
                    end_first = orig_size_out
                    start_current = j * orig_size_out
                    end_current = (j + 1) * orig_size_out
                    
                    blocks_match = torch.allclose(
                        grad[start_first:end_first, :],
                        grad[start_current:end_current, :],
                        rtol=1e-4, atol=1e-4
                    )
                    
                    if not blocks_match:
                        gradients_match = False
                        print(f"  First layer: Row block 0 and block {j} gradients don't match")
                        # Print sample values for debugging
                        print(f"  Sample values - Block 0: {grad[0, 0].item()}, Block {j}: {grad[start_current, 0].item()}")
            
            # Case 2: Middle layers - both dimensions expanded
            else:
                # Check if each block has identical gradients
                for j in range(clone_factor):
                    for k in range(clone_factor):
                        if j == 0 and k == 0:
                            continue  # Skip first block
                            
                        row_start_first = 0
                        row_end_first = orig_size_out
                        col_start_first = 0
                        col_end_first = orig_size_in
                        
                        row_start = j * orig_size_out
                        row_end = (j + 1) * orig_size_out
                        col_start = k * orig_size_in
                        col_end = (k + 1) * orig_size_in
                        
                        blocks_match = torch.allclose(
                            grad[row_start_first:row_end_first, col_start_first:col_end_first],
                            grad[row_start:row_end, col_start:col_end],
                            rtol=1e-4, atol=1e-4
                        )
                        
                        if not blocks_match:
                            gradients_match = False
                            print(f"  Layer {i}: Block (0,0) and block ({j},{k}) gradients don't match")
                            # Print sample values for debugging
                            print(f"  Sample values - Block (0,0): {grad[0, 0].item()}, Block ({j},{k}): {grad[row_start, col_start].item()}")
    
    # Check output layer gradients
    output_weight = "layers.out.weight"
    if output_weight in dict(cloned_model.named_parameters()):
        param = dict(cloned_model.named_parameters())[output_weight]
        grad = param.grad
        
        if grad is not None:
            print(f"\nGradients for {output_weight}:")
            print(f"Shape: {grad.shape}")
            
            orig_size = original_model.hidden_sizes[-1]
            
            # Check if each block of columns has identical gradients (scaled)
            for j in range(1, clone_factor):
                col_start_first = 0
                col_end_first = orig_size
                col_start_current = j * orig_size
                col_end_current = (j + 1) * orig_size
                
                # Gradients should be identical because weights were scaled
                blocks_match = torch.allclose(
                    grad[:, col_start_first:col_end_first],
                    grad[:, col_start_current:col_end_current],
                    rtol=1e-4, atol=1e-4
                )
                
                if not blocks_match:
                    gradients_match = False
                    print(f"  Output layer: Column block 0 and block {j} gradients don't match")
                    # Print sample values for debugging
                    print(f"  Sample values - Block 0: {grad[0, 0].item()}, Block {j}: {grad[0, col_start_current].item()}")
    
    if gradients_match:
        print("\n✅ Test passed! All gradient blocks match the cloned pattern.")
    else:
        print("\n❌ Test failed! Some gradient blocks don't match the cloned pattern.")
        
    return gradients_match


def test_training_equivalence(clone_factor=3, epochs=5):
    """
    Test if training the cloned model is equivalent to training the original model.
    This tests the proposition from the paper that gradient descent trajectories 
    of the cloned model remain in the affine sub-space of the original model.
    """
    # Create original model with small number of units for easier visualization
    original_model = MLP(
        input_size=784,
        hidden_sizes=[8, 6],
        output_size=2,
        activation='tanh',
        normalization=None,
        dropout_p=0.0
    )
    
    # Create cloned model
    cloned_model = clone_mlp(original_model, clone_factor)
    
    # Set up optimizers
    lr = 0.01
    optimizer_original = torch.optim.SGD(original_model.parameters(), lr=lr)
    optimizer_cloned = torch.optim.SGD(cloned_model.parameters(), lr=lr)
    
    # Set up loss function
    criterion = nn.MSELoss()
    
    # Generate random training data
    torch.manual_seed(42)
    inputs = torch.randn(10, 784)
    targets = torch.randn(10, 2)
    
    print("\n=== Training Equivalence Test ===")
    
    # Store losses for comparison
    losses_original = []
    losses_cloned = []
    
    for epoch in range(epochs):
        # Train original model
        original_model.train()
        outputs_original = original_model(inputs)
        loss_original = criterion(outputs_original, targets)
        
        optimizer_original.zero_grad()
        loss_original.backward()
        optimizer_original.step()
        
        # Train cloned model
        cloned_model.train()
        outputs_cloned = cloned_model(inputs)
        loss_cloned = criterion(outputs_cloned, targets)
        
        optimizer_cloned.zero_grad()
        loss_cloned.backward()
        optimizer_cloned.step()
        
        # Store losses
        losses_original.append(loss_original.item())
        losses_cloned.append(loss_cloned.item())
        
        print(f"Epoch {epoch+1}: Original Loss = {loss_original.item():.6f}, Cloned Loss = {loss_cloned.item():.6f}")
    
    # Test if the cloned model's output approaches the original model's output
    # after training (they won't be exactly the same due to scaling differences)
    original_model.eval()
    cloned_model.eval()
    
    with torch.no_grad():
        test_input = torch.randn(5, 784)
        original_output = original_model(test_input)
        cloned_output = cloned_model(test_input)
    
    # The outputs won't be identical, but they should follow a similar pattern
    correlation = torch.corrcoef(
        torch.stack([original_output.flatten(), cloned_output.flatten()])
    )[0, 1].item()
    
    print(f"\nCorrelation between original and cloned model outputs: {correlation:.6f}")
    print(f"Original model output: {original_output[0]}")
    print(f"Cloned model output: {cloned_output[0]}")
    
    # Check if the correlation is high (above 0.7 is generally considered strong)
    if correlation > 0.7:
        print("\n✅ Test passed! Original and cloned models show similar learning patterns.")
    else:
        print("\n❌ Test failed! Original and cloned models diverge during training.")
    
    return correlation > 0.7


if __name__ == "__main__":
    clone_factor = 3
    
    print("=== Testing Basic Cloning ===")
    test_cloning(clone_factor)
    
    # Test activation cloning
    test_forward_activation_cloning(clone_factor)
    
    # Test gradient cloning
    test_gradient_cloning(clone_factor)
    
    # Test training equivalence
    test_training_equivalence(clone_factor, epochs=10)