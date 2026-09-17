import sqlite3
import torch
from safetensors.torch import save_file
import struct
import os
import argparse
from pathlib import Path
import numpy as np

# Draw Things writes .ckpt files through s4nnc / ccv. The high 32 bits of the
# `type` column hold the codec identifier used to encode the `data` blob, and the
# `datatype` column carries CCV_QX (0x40000) for palette-quantized tensors.
# Identifiers taken from liuliu/s4nnc nnc/Store.swift.
CODEC_IDENTIFIERS = {
    0x217: "zip",
    0x511: "ezm7",
    0xF7217: "fpzip",
    0x8A1E4B: "q4p",
    0x8A1E5B: "q5p",
    0x8A1E6B: "q6p",
    0x8A1E7B: "q7p",
    0x8A1E8B: "q8p",
    0x8A1E9B: "i8x",
    0x8A1EAB: "i8x (Q4_K)",
    0x8A1EAC: "i8x (Q3_K)",
    0x8A1EAD: "i8x (Q2_K)",
    0x8A1EAE: "i8x (IQ2_S)",
    0x8A1EAF: "i8x (IQ2_XS)",
    0x8A1EB0: "i8x (IQ3_S)",
    0x8A1EB1: "i8x (IQ3_XXS)",
    0x8A1EB2: "i8x (IQ2_XXS)",
    0x8A1EB3: "i8x (Q5_K)",
    0x8A1EB4: "i8x (Q6_K)",
}
CCV_DATA_TYPE_MASK = 0xFF000
CCV_QX = 0x40000
# Bit 28 of the identifier means the bytes live in the sibling `<file>-tensordata`
# file; the blob is then two little-endian uint64: offset and length.
EXTERNAL_DATA_FLAG = 0x10000000
CODEC_ID_MASK = 0x0FFFFFFF

def convert_ckpt_to_safetensors(ckpt_file, overwrite=False, remove_ckpt=False, verbose=False):
    """Convert a single Draw Things .ckpt file to .safetensors format.

    `remove_ckpt` is accepted for backwards compatibility but ignored: the
    original file is never deleted because the conversion cannot be verified
    as lossless. Returns True, "partial" (some tensors skipped) or False.
    """
    print(f"\n{'='*60}")
    print(f"Converting: {ckpt_file}")
    print(f"{'='*60}")
    
    # Create output filename (same location as input)
    output_file = ckpt_file.replace(".ckpt", ".safetensors")
    
    # Check if output already exists
    if os.path.exists(output_file) and not overwrite:
        file_size = os.path.getsize(output_file) / (1024*1024)
        print(f"⊘ Skipping - output already exists ({file_size:.2f} MB)")
        print(f"  Use --overwrite to replace existing file")
        return False
    
    try:
        # Check if this looks like a quantized model from filename
        quantized_indicators = ['q4', 'q6', 'q8', 'quantized', 'compressed']
        is_quantized_file = any(indicator in ckpt_file.lower() for indicator in quantized_indicators)
        
        # Connect to the SQLite database
        conn = sqlite3.connect(ckpt_file)
        cursor = conn.cursor()
        
        # Check if the tensors table exists
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tensors'")
            if not cursor.fetchone():
                conn.close()
                print(f"✗ Error: This file does not appear to be a valid Draw Things .ckpt file.")
                print(f"  The file is missing the 'tensors' table. It may be a different format.")
                return False
        except sqlite3.Error as e:
            conn.close()
            print(f"✗ Error: Failed to read database structure: {e}")
            print(f"  This file may not be a valid Draw Things .ckpt file.")
            return False
        
        # Get all tensors
        cursor.execute("SELECT name, type, format, datatype, dim, data FROM tensors")
        rows = cursor.fetchall()
        
        print(f"Found {len(rows)} tensors")
        
        state_dict = {}
        failed_tensors = []
        quantized_tensors = []  # Track quantized tensors separately for summary
        
        for idx, row in enumerate(rows):
            name, tensor_type, format_type, datatype, dim_blob, data_blob = row
            
            # Refuse encoded blobs instead of guessing. Draw Things compresses or
            # quantizes most tensors (ezm7, fpzip, q6p/q8p, ...). Reinterpreting
            # those bytes as raw floats produces garbage that still "converts".
            identifier = ((tensor_type or 0) >> 32) & 0xFFFFFFFF
            codec_id = identifier & CODEC_ID_MASK
            is_external = bool(identifier & EXTERNAL_DATA_FLAG)
            datatype_bits = (datatype or 0) & CCV_DATA_TYPE_MASK
            if codec_id != 0 or is_external or datatype_bits == CCV_QX:
                reasons = []
                if codec_id != 0:
                    reasons.append(f"encoded with {CODEC_IDENTIFIERS.get(codec_id, f'unknown codec 0x{codec_id:x}')}")
                elif datatype_bits == CCV_QX:
                    reasons.append("palette-quantized")
                if is_external:
                    reasons.append("bytes live in the external -tensordata file")
                print(f"  Skipping '{name}': {'; '.join(reasons)}. This tool cannot decode it.")
                quantized_tensors.append(name)
                failed_tensors.append(name)
                continue
            if data_blob is None:
                print(f"  Skipping '{name}': no inline data (probably stored in the external -tensordata file).")
                failed_tensors.append(name)
                continue
            
            # Clean up tensor name - remove Draw Things specific prefixes/suffixes
            # Draw Things might wrap names like "__text_model__[t-8-0]__up__"
            # We need to extract the actual parameter name
            clean_name = name
            if clean_name.startswith("__") and "__" in clean_name[2:]:
                # Extract the actual name between __ markers
                parts = clean_name.strip("_").split("__")
                # Look for the actual parameter name (usually the middle part)
                for part in parts:
                    if not part.startswith("[") and part:
                        clean_name = part
                        break
            
            # Parse dimensions from blob - remove trailing zeros only
            if dim_blob:
                all_dims = struct.unpack(f'{len(dim_blob)//4}i', dim_blob)
                # Remove only trailing zeros, not zeros in the middle
                dims = list(all_dims)
                while dims and dims[-1] == 0:
                    dims.pop()
                dims = tuple(dims)
            else:
                dims = ()
            
            # Convert blob to bytes
            tensor_data = bytes(data_blob)
            data_size = len(tensor_data)
            
            # Infer PyTorch dtype from data size and dimensions
            if dims:
                # Check if any dimension is zero (would result in zero elements)
                if any(d == 0 for d in dims):
                    print(f"  Warning: Tensor '{name}' has zero dimension(s) in shape {dims}. Skipping empty tensor.")
                    failed_tensors.append(name)
                    continue
                
                expected_elements = 1
                for d in dims:
                    expected_elements *= d
                
                # Sanity check: expected_elements should never be 0 at this point
                if expected_elements == 0:
                    print(f"  Error: Tensor '{name}' has invalid dimensions {dims} resulting in 0 elements.")
                    failed_tensors.append(name)
                    continue
                
                # Calculate bytes per element with tolerance for floating point errors
                bytes_per_element = data_size / expected_elements
                
                # Use rounding to handle floating point precision issues
                if abs(bytes_per_element - 4.0) < 0.1:
                    dtype = torch.float32
                    dtype_size = 4
                elif abs(bytes_per_element - 2.0) < 0.1:
                    dtype = torch.float16
                    dtype_size = 2
                elif abs(bytes_per_element - 1.0) < 0.1:
                    dtype = torch.uint8
                    dtype_size = 1
                elif abs(bytes_per_element - 8.0) < 0.1:
                    dtype = torch.float64
                    dtype_size = 8
                else:
                    # Fallback: try to infer from actual data size
                    dtype = torch.float32
                    dtype_size = 4
                
                # Validate data size matches expected size
                expected_bytes = expected_elements * dtype_size
                if data_size != expected_bytes:
                    # Check if the difference is small (might be padding/alignment)
                    diff = abs(data_size - expected_bytes)
                    if diff <= dtype_size and data_size > expected_bytes:
                        # Likely just extra padding, truncate
                        print(f"  Warning: Tensor '{name}' has {diff} extra bytes (likely padding). Truncating...")
                        tensor_data = tensor_data[:expected_bytes]
                    elif data_size < expected_bytes:
                        # Check if this looks like a quantized model (data much smaller than expected)
                        size_ratio = data_size / expected_bytes if expected_bytes > 0 else 0
                        if size_ratio < 0.1:  # Less than 10% of expected size
                            quantized_tensors.append(name)
                            failed_tensors.append(name)
                            continue
                        else:
                            # Data is too short but not obviously quantized
                            print(f"  Error: Tensor '{name}' is too short. Expected {expected_bytes} bytes, got {data_size}.")
                            failed_tensors.append(name)
                            continue
                    else:
                        # Significant size mismatch - try to recalculate dtype
                        print(f"  Warning: Tensor '{name}' size mismatch. Expected {expected_bytes} bytes, got {data_size}. Recalculating dtype...")
                        # Try to find a better dtype match
                        alt_bytes_per_element = data_size / expected_elements
                        dims_updated = False
                        
                        if abs(alt_bytes_per_element - 4.0) < 0.1:
                            dtype = torch.float32
                            dtype_size = 4
                        elif abs(alt_bytes_per_element - 2.0) < 0.1:
                            dtype = torch.float16
                            dtype_size = 2
                        elif abs(alt_bytes_per_element - 1.0) < 0.1:
                            dtype = torch.uint8
                            dtype_size = 1
                        elif abs(alt_bytes_per_element - 8.0) < 0.1:
                            dtype = torch.float64
                            dtype_size = 8
                        else:
                            # Try to infer from data size being a multiple of common dtype sizes
                            if data_size % 4 == 0:
                                # Might be float32 with wrong dimensions
                                actual_elements = data_size // 4
                                if actual_elements > 0:
                                    dtype = torch.float32
                                    dtype_size = 4
                                    print(f"  Attempting float32 with {actual_elements} elements (dimensions may be incorrect)")
                                    # Update dimensions to match actual data
                                    dims = (actual_elements,)
                                    expected_bytes = data_size
                                    dims_updated = True
                                else:
                                    print(f"  Error: Cannot determine dtype for tensor '{name}'")
                                    failed_tensors.append(name)
                                    continue
                            elif data_size % 2 == 0:
                                actual_elements = data_size // 2
                                if actual_elements > 0:
                                    dtype = torch.float16
                                    dtype_size = 2
                                    print(f"  Attempting float16 with {actual_elements} elements (dimensions may be incorrect)")
                                    dims = (actual_elements,)
                                    expected_bytes = data_size
                                    dims_updated = True
                                else:
                                    print(f"  Error: Cannot determine dtype for tensor '{name}'")
                                    failed_tensors.append(name)
                                    continue
                            else:
                                print(f"  Error: Cannot determine dtype for tensor '{name}' (size: {data_size} bytes, expected: {expected_bytes} bytes)")
                                failed_tensors.append(name)
                                continue
                        
                        if not dims_updated:
                            expected_bytes = expected_elements * dtype_size
                            if data_size != expected_bytes:
                                # Final attempt: check if we can use the actual data size
                                actual_elements = data_size // dtype_size
                                if actual_elements * dtype_size == data_size and actual_elements > 0:
                                    print(f"  Using actual data size: {actual_elements} elements (dimensions may be incorrect)")
                                    dims = (actual_elements,)
                                    expected_bytes = data_size
                                else:
                                    print(f"  Error: Still size mismatch after dtype recalculation for '{name}'")
                                    failed_tensors.append(name)
                                    continue
            else:
                # Scalar or unknown shape - default to float32
                dtype = torch.float32
                dtype_size = 4
                # Ensure data size is a multiple of dtype size
                if data_size % dtype_size != 0:
                    # Pad to align
                    padding = dtype_size - (data_size % dtype_size)
                    tensor_data = tensor_data + b'\x00' * padding
            
            # Convert bytes to tensor using numpy as intermediate (more robust)
            try:
                # Map torch dtype to numpy dtype
                dtype_map = {
                    torch.float32: np.float32,
                    torch.float16: np.float16,
                    torch.float64: np.float64,
                    torch.uint8: np.uint8,
                    torch.int8: np.int8,
                    torch.int16: np.int16,
                    torch.int32: np.int32,
                    torch.int64: np.int64,
                }
                np_dtype = dtype_map.get(dtype, np.float32)
                
                # Create a writable copy of the buffer to avoid warnings
                tensor_data_copy = bytearray(tensor_data)
                # Create numpy array from bytes
                np_array = np.frombuffer(tensor_data_copy, dtype=np_dtype)
                # Convert to PyTorch tensor (copy ensures it's writable)
                tensor = torch.from_numpy(np_array.copy()).clone()
            except (ValueError, TypeError) as e:
                # Fallback: try torch.frombuffer with a writable buffer
                try:
                    # Create a writable copy to avoid warnings
                    tensor_data_copy = bytearray(tensor_data)
                    tensor = torch.frombuffer(tensor_data_copy, dtype=dtype).clone()
                except Exception as e2:
                    print(f"  Error creating tensor '{name}': {e2}")
                    failed_tensors.append(name)
                    continue
            
            # Reshape if dimensions are specified
            if dims:
                try:
                    tensor = tensor.reshape(dims)
                except RuntimeError as e:
                    print(f"  Error reshaping tensor '{name}' to {dims}: {e}")
                    # Check element counts for diagnostic purposes
                    total_elements = tensor.numel()
                    expected_elements = 1
                    for d in dims:
                        expected_elements *= d
                    if total_elements != expected_elements:
                        print(f"  Warning: Reshape failed for '{name}'. Expected {expected_elements} elements, got {total_elements}")
                    else:
                        print(f"  Warning: Reshape failed for '{name}' despite matching element count ({total_elements}). This may indicate an incompatible shape.")
                    # Always skip the tensor if reshape fails, regardless of element count
                    failed_tensors.append(name)
                    continue
            
            state_dict[name] = tensor
            
            # Show progress every 100 tensors
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(rows)} tensors...")
        
        conn.close()
        
        # Check if we have any tensors to save
        if not state_dict:
            if quantized_tensors and len(quantized_tensors) == len(rows):
                print(f"  ✗ Error: every tensor in this file is compressed or quantized.")
                print(f"  All {len(rows)} tensors use a Draw Things codec (ezm7/fpzip/q6p/q8p/...) this converter cannot decode.")
                raise ValueError("Fully encoded model - no tensors can be extracted")
            else:
                raise ValueError("No tensors were successfully extracted from the .ckpt file")
        
        # Report any failed tensors
        if failed_tensors:
            # Separate quantized tensors from other failures
            other_failed = [t for t in failed_tensors if t not in quantized_tensors]
            
            # Report quantized tensors (if any)
            if quantized_tensors:
                print(f"  ⚠ Warning: {len(quantized_tensors)} tensor(s) are quantized/compressed and were skipped.")
                if is_quantized_file:
                    print(f"  This file appears to be a quantized model (q4p/q6p/q8p format).")
                print(f"  Quantized models use compressed storage and are not fully supported.")
                if len(quantized_tensors) <= 5:
                    print(f"  Skipped quantized tensors: {', '.join(quantized_tensors)}")
                else:
                    print(f"  Skipped quantized tensors (first 5): {', '.join(quantized_tensors[:5])}")
                    print(f"  ... and {len(quantized_tensors) - 5} more quantized tensors")
            
            # Report other failures
            if other_failed:
                print(f"  Warning: {len(other_failed)} tensor(s) failed to convert: {', '.join(other_failed[:10])}")
                if len(other_failed) > 10:
                    print(f"  ... and {len(other_failed) - 10} more")
            
            # Summary if we have both types
            if quantized_tensors and other_failed:
                print(f"  Total: {len(failed_tensors)} failed, {len(state_dict)} succeeded out of {len(rows)} tensors")
        
        # Detect file type based on tensor name patterns
        tensor_names = sorted(state_dict.keys())
        file_type = detect_file_type(tensor_names, ckpt_file)
        print(f"\n  Detected file type: {file_type}")
        
        # Diagnostic: Show tensor name patterns to help debug model detection issues
        if verbose:
            print(f"\n  All tensor names ({len(tensor_names)} total):")
            for name in tensor_names:
                print(f"    {name}")
        else:
            print(f"\n  Tensor name patterns (first 10):")
            for name in tensor_names[:10]:
                print(f"    {name}")
            if len(tensor_names) > 10:
                print(f"  ... and {len(tensor_names) - 10} more tensors")
                print(f"  Tensor name patterns (last 5):")
                for name in tensor_names[-5:]:
                    print(f"    {name}")
        
        # Type-specific validation and key detection
        if "LoRA" in file_type:
            # LoRAs should have __down__ and __up__ pairs
            down_tensors = [n for n in tensor_names if '__down__' in n]
            up_tensors = [n for n in tensor_names if '__up__' in n]
            if down_tensors and up_tensors:
                print(f"  LoRA structure: {len(down_tensors)} down tensors, {len(up_tensors)} up tensors")
                if len(down_tensors) != len(up_tensors):
                    print(f"  ⚠ Warning: Mismatch between down ({len(down_tensors)}) and up ({len(up_tensors)}) tensors")
        elif "VAE" in file_type:
            # VAEs should have encoder and decoder components
            encoder_tensors = [n for n in tensor_names if '__encoder__' in n]
            decoder_tensors = [n for n in tensor_names if '__decoder__' in n]
            if encoder_tensors and decoder_tensors:
                print(f"  VAE structure: {len(encoder_tensors)} encoder tensors, {len(decoder_tensors)} decoder tensors")
        elif "Encoder" in file_type:
            # Text/audio encoders
            text_tensors = [n for n in tensor_names if '__text_model__' in n]
            if text_tensors:
                print(f"  Encoder structure: {len(text_tensors)} text model tensors")
        
        # Check for common model detection keys that ComfyUI might look for
        common_keys = [
            'model.diffusion_model', 'model_ema', 'state_dict', 
            'model', 'unet', 'diffusion_model', 'first_stage_model',
            'cond_stage_model', 'text_encoder', 'vae'
        ]
        found_keys = [key for key in common_keys if any(k.startswith(key) or key in k for k in tensor_names)]
        if found_keys:
            print(f"  Found potential model keys: {', '.join(found_keys)}")
        elif "LoRA" not in file_type and "VAE" not in file_type and "Encoder" not in file_type:
            # Only warn if it's not a known type that uses non-standard naming
            print(f"  ⚠ Warning: No common model detection keys found. Model type detection may fail.")
            print(f"  This might indicate the model uses a non-standard naming convention.")
        
        # Save as safetensors with metadata
        print(f"\nSaving to {output_file}...")
        metadata = {"format": "pt"}  # PyTorch format metadata
        save_file(state_dict, output_file, metadata=metadata)
        
        # Verify the file was written correctly
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file was not created: {output_file}")
        
        file_size = os.path.getsize(output_file)
        if file_size == 0:
            raise ValueError(f"Output file is empty: {output_file}")
        
        # Try to verify the safetensors file can be read
        try:
            from safetensors import safe_open
            with safe_open(output_file, framework="pt") as f:
                keys = list(f.keys())
                if len(keys) != len(state_dict):
                    raise ValueError(f"File verification failed: expected {len(state_dict)} tensors, found {len(keys)}")
        except Exception as e:
            print(f"  Warning: Could not verify safetensors file: {e}")
            # Don't fail the conversion, but warn the user
        
        # Deleting the source file is intentionally not supported: this tool cannot
        # prove the output is a faithful copy, so the original must stay.
        if failed_tensors:
            print(f"⚠ Partial conversion: saved {len(state_dict)} of {len(rows)} tensors ({file_size / (1024*1024):.2f} MB)")
            print(f"  The .safetensors file is NOT a faithful copy of the .ckpt. Keep the original.")
            return "partial"
        
        print(f"✓ Successfully converted! Saved {len(state_dict)} tensors ({file_size / (1024*1024):.2f} MB)")
        return True
        
    except Exception as e:
        print(f"✗ Error converting {ckpt_file}: {e}")
        return False

def detect_file_type(tensor_names, ckpt_file):
    """Detect the type of model file based on tensor name patterns."""
    # Convert to lowercase for case-insensitive matching
    names_lower = [name.lower() for name in tensor_names]
    file_lower = ckpt_file.lower()
    
    # Check for LoRA patterns (has __down__ and __up__ suffixes)
    has_lora_patterns = any('__down__' in name or '__up__' in name for name in names_lower)
    
    # Check for VAE patterns (encoder/decoder)
    has_vae_patterns = any('__encoder__' in name or '__decoder__' in name for name in names_lower)
    
    # Check for CLIP/text encoder patterns
    has_clip_patterns = any('__text_model__' in name and '__down__' not in name and '__up__' not in name 
                            for name in names_lower)
    
    # Check for diffusion model patterns (DIT, UNET, etc.)
    has_diffusion_patterns = any('__dit__' in name or '__unet__' in name or '__unet_fixed__' in name 
                                 for name in names_lower)
    
    # Determine type based on patterns
    if has_lora_patterns:
        # Further classify LoRA type
        if any('__dit__' in name for name in names_lower):
            return "LoRA (Flux/DIT)"
        elif any('__unet__' in name or '__unet_fixed__' in name for name in names_lower):
            return "LoRA (SDXL/UNET)"
        elif any('__text_model__' in name for name in names_lower):
            return "LoRA (CLIP/Text Encoder)"
        else:
            return "LoRA"
    elif has_vae_patterns:
        return "VAE"
    elif has_clip_patterns:
        # Check if it's an audio encoder (T5, UMT5, etc.)
        if any('t5' in name or 'umt5' in name for name in names_lower) or 'encoder' in file_lower:
            return "Audio Encoder"
        else:
            return "CLIP/Text Encoder"
    elif has_diffusion_patterns:
        if any('__dit__' in name for name in names_lower):
            return "Diffusion Model (Flux/DIT)"
        else:
            return "Diffusion Model (SDXL/UNET)"
    else:
        # Try to infer from filename
        if 'lora' in file_lower:
            return "LoRA (Unknown type)"
        elif 'vae' in file_lower:
            return "VAE"
        elif 'clip' in file_lower or 'encoder' in file_lower:
            return "Encoder"
        else:
            return "Unknown/Other"

def find_ckpt_files(folder_path):
    """Recursively find all .ckpt files in folder and subfolders."""
    ckpt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.ckpt'):
                ckpt_files.append(os.path.join(root, file))
    return ckpt_files

def inspect_safetensors(safetensors_file):
    """Inspect a safetensors file and show tensor names and metadata."""
    try:
        from safetensors import safe_open
        print(f"\n{'='*60}")
        print(f"Inspecting: {safetensors_file}")
        print(f"{'='*60}")
        
        with safe_open(safetensors_file, framework="pt") as f:
            keys = list(f.keys())
            print(f"Total tensors: {len(keys)}")
            print(f"\nTensor names (first 20):")
            for i, key in enumerate(keys[:20]):
                tensor = f.get_tensor(key)
                print(f"  {i+1}. {key} - shape: {tensor.shape}, dtype: {tensor.dtype}")
            
            if len(keys) > 20:
                print(f"\n  ... and {len(keys) - 20} more tensors")
                print(f"\nTensor names (last 5):")
                for key in keys[-5:]:
                    tensor = f.get_tensor(key)
                    print(f"  {key} - shape: {tensor.shape}, dtype: {tensor.dtype}")
            
            # Check for metadata
            metadata = f.metadata()
            if metadata:
                print(f"\nMetadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
            else:
                print(f"\nNo metadata found in file")
                
    except Exception as e:
        print(f"Error inspecting file: {e}")

def main():
    parser = argparse.ArgumentParser(
        description='Convert Draw Things .ckpt files to .safetensors format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all .ckpt files in a folder (including subfolders)
  python convert_ckpt_to_safetensors.py --folder "c:/models"
  
  # Convert a single file
  python convert_ckpt_to_safetensors.py --file "c:/models/my_lora.ckpt"
  
  # Convert with overwrite
  python convert_ckpt_to_safetensors.py --folder "c:/models" --overwrite
  
  # Convert with verbose output (show all tensor names)
  python convert_ckpt_to_safetensors.py --file "c:/models/my_lora.ckpt" --verbose
  
  # Inspect an existing safetensors file
  python convert_ckpt_to_safetensors.py --inspect "c:/models/my_model.safetensors"
        """
    )
    
    # Create mutually exclusive group for --folder, --file, and --inspect
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--folder', type=str, help='Folder to scan for .ckpt files (includes subfolders)')
    group.add_argument('--file', type=str, help='Single .ckpt file to convert')
    group.add_argument('--inspect', type=str, help='Inspect an existing .safetensors file (show tensor names and metadata)')
    
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite existing .safetensors files')
    parser.add_argument('--remove-ckpt', action='store_true',
                       help='(disabled) Kept for compatibility. Originals are never deleted '
                            'because the conversion cannot be verified as lossless.')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show all tensor names during conversion (for debugging)')
    
    args = parser.parse_args()
    
    # Handle inspect mode
    if args.inspect:
        if not os.path.exists(args.inspect):
            print(f"Error: File not found: {args.inspect}")
            return
        if not args.inspect.endswith('.safetensors'):
            print(f"Error: File must have .safetensors extension")
            return
        inspect_safetensors(args.inspect)
        return
    
    print("Draw Things .ckpt to .safetensors Converter")
    print("=" * 60)
    
    # Determine files to convert
    if args.folder:
        if not os.path.exists(args.folder):
            print(f"Error: Folder not found: {args.folder}")
            return
        
        print(f"Scanning folder: {args.folder}")
        files_to_convert = find_ckpt_files(args.folder)
        
        if not files_to_convert:
            print(f"No .ckpt files found in {args.folder}")
            return
        
        print(f"Found {len(files_to_convert)} .ckpt file(s)\n")
    
    else:  # args.file
        if not os.path.exists(args.file):
            print(f"Error: File not found: {args.file}")
            return
        
        if not args.file.endswith('.ckpt'):
            print(f"Error: File must have .ckpt extension")
            return
        
        files_to_convert = [args.file]
    
    # Display options
    print(f"Options:")
    print(f"  Overwrite existing: {args.overwrite}")
    if args.remove_ckpt:
        print(f"  ⚠ --remove-ckpt is disabled: originals are never deleted")
    print(f"  Verbose mode: {args.verbose}")
    
    # Convert files
    successful = 0
    partial = 0
    skipped = 0
    failed = 0
    
    for ckpt_file in files_to_convert:
        output_file = ckpt_file.replace(".ckpt", ".safetensors")
        file_existed_before = os.path.exists(output_file)
        
        result = convert_ckpt_to_safetensors(ckpt_file, args.overwrite, args.remove_ckpt, args.verbose)
        
        # Check if output file exists after conversion to distinguish skip vs failure
        file_exists_after = os.path.exists(output_file)
        
        if result == "partial":
            partial += 1
        elif result:
            successful += 1
        elif file_exists_after and not args.overwrite:
            # File was skipped because it exists and overwrite is False
            # (convert_ckpt_to_safetensors returns False in this case)
            skipped += 1
        else:
            # Conversion failed (file doesn't exist or other error occurred)
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files: {len(files_to_convert)}")
    print(f"✓ Successfully converted: {successful}")
    print(f"⚠ Partial (tensors missing, keep originals): {partial}")
    print(f"⊘ Skipped (already exists): {skipped}")
    print(f"✗ Failed: {failed}")

if __name__ == "__main__":
    main()