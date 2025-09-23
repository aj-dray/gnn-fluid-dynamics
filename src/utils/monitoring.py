"""
Monitoring utilities for training metrics and model diagnostics
"""

import torch


class ModelMonitor:
    """Utility class for monitoring model parameters and gradients during training"""
    
    def __init__(self):
        self._face_mlp_old_params = None
    
    def monitor_face_mlp_gradients(self, face_mlp, logger, step):
        """Monitor gradients for each output channel of decoder.face_mlp"""
        # Find the final linear layer (output layer) in the face_mlp
        output_layer = None
        for layer in reversed(face_mlp):
            if isinstance(layer, torch.nn.Linear):
                output_layer = layer
                break
        
        if output_layer is not None and output_layer.weight.grad is not None:
            # Store parameters before update for computing change
            old_weight = output_layer.weight.data.clone()
            old_bias = output_layer.bias.data.clone() if output_layer.bias is not None else None
            
            # Get gradients for each output channel
            grad_weight = output_layer.weight.grad  # shape: [out_features, in_features]
            grad_bias = output_layer.bias.grad if output_layer.bias is not None else None
            
            # Calculate gradient norms per output channel
            for ch in range(grad_weight.shape[0]):
                # Weight gradient norm for this channel
                weight_grad_norm = torch.norm(grad_weight[ch]).item()
                logger.save_scalar(weight_grad_norm, step=step, 
                                 prefix=f"gradients/face_mlp_weight_ch{ch}")
                
                # Bias gradient norm for this channel (if bias exists)
                if grad_bias is not None:
                    bias_grad_norm = abs(grad_bias[ch].item())
                    logger.save_scalar(bias_grad_norm, step=step,
                                     prefix=f"gradients/face_mlp_bias_ch{ch}")
            
            # Store old parameters for update monitoring
            self._face_mlp_old_params = (old_weight, old_bias)
    
    def monitor_face_mlp_updates(self, face_mlp, logger, step):
        """Monitor parameter updates for each output channel of decoder.face_mlp"""
        if self._face_mlp_old_params is None:
            return
            
        # Find the final linear layer (output layer) in the face_mlp
        output_layer = None
        for layer in reversed(face_mlp):
            if isinstance(layer, torch.nn.Linear):
                output_layer = layer
                break
        
        if output_layer is not None:
            old_weight, old_bias = self._face_mlp_old_params
            
            # Calculate updates per channel
            weight_update = output_layer.weight.data - old_weight
            bias_update = output_layer.bias.data - old_bias if old_bias is not None else None
            
            for ch in range(weight_update.shape[0]):
                # Weight update norm for this channel
                weight_update_norm = torch.norm(weight_update[ch]).item()
                logger.save_scalar(weight_update_norm, step=step,
                                 prefix=f"updates/face_mlp_weight_ch{ch}")
                
                # Bias update for this channel (if bias exists)
                if bias_update is not None:
                    bias_update_val = abs(bias_update[ch].item())
                    logger.save_scalar(bias_update_val, step=step,
                                     prefix=f"updates/face_mlp_bias_ch{ch}")
            
            # Clean up
            self._face_mlp_old_params = None

    def monitor_scalar_parameters(self, model, logger, step):
        """Monitor all scalar parameters in the model (torch.nn.Parameter with single element)"""
        for name, param in model.named_parameters():
            # Check if parameter is scalar (single element)
            if isinstance(param, torch.nn.Parameter) and param.numel() == 1:
                # Clean parameter name for logging (replace dots with underscores)
                clean_name = name.replace('.', '_')
                
                # Log the parameter value
                logger.save_scalar(param.item(), step=step,
                                 prefix=f"scalar_params/{clean_name}")
                
                # Log the gradient if it exists
                if param.grad is not None:
                    logger.save_scalar(param.grad.item(), step=step,
                                     prefix=f"scalar_grads/{clean_name}")
