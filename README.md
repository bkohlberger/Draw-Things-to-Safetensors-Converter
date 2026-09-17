# Draw Things to Safetensors Converter

> ## ⚠️ Archived research. Do not rely on this tool.
>
> This was a November 2025 experiment and is no longer maintained. Before archiving it was checked against the real Draw Things on-disk format, and it does **not** solve the problem:
>
> - **Tensor names stay in Draw Things' internal scheme** (`__unet__[t-0-0]__down__`). ComfyUI, Automatic1111 and diffusers cannot load them. Mapping them back needs Draw Things' per-architecture model tables.
> - **Most real Draw Things files are compressed or quantized** (ezm7, fpzip, q6p/q8p, or stored in an external `-tensordata` file). Those tensors are skipped with a message. Only plain float16/float32 tensors are written.
> - **Draw Things exports natively.** The app can export LoRAs you trained to standard safetensors (SD 1.x/2.x, SDXL, SSD-1B, Flux.1, Flux.2 klein, Qwen Image, Z Image, ERNIE Image, Cosmos 2.5, per `FeaturesMatrix.swift` in [draw-things-community](https://github.com/drawthingsai/draw-things-community)). Use that instead.
> - **`--remove-ckpt` is disabled.** Originals are never deleted.
>
> Kept public as a record of the SQLite layout, nothing more.

A Python tool to convert Draw Things `.ckpt` model files back to standard `.safetensors` format for use with ComfyUI, Automatic1111, and other AI image generation tools.

## Background

Draw Things stores AI models in a proprietary SQLite-based `.ckpt` format that is incompatible with most other AI tools. This converter extracts the tensor data from Draw Things' database format and saves it as standard `.safetensors` files that can be used across different platforms.

## Features

- ✅ Batch convert all `.ckpt` files in a folder
- ✅ Convert single files on demand
- ✅ Command-line interface (`--folder`, `--file`, `--inspect`)
- ✅ Automatically skip already-converted files
- ✅ Preserves shapes and Draw Things' internal tensor names (not remapped)
- ⛔ Compressed or quantized tensors are skipped, not decoded
- ✅ Progress tracking for large conversions
- ✅ Error handling with detailed reporting

## Requirements

- Python 3.7+
- PyTorch
- safetensors

## Installation

1. Clone this repository:
```bash
git clone https://github.com/EctoSpace/Draw-Things-to-Safetensors-Converter.git
cd Draw-Things-to-Safetensors-Converter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or using a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Basic Usage

The converter uses command-line arguments for easy and flexible operation.

#### Convert All Files in a Folder

Scans the folder and all subfolders for `.ckpt` files and converts them in place:

```bash
python convert_ckpt_to_safetensors.py --folder "c:/models"
```

On macOS/Linux:
```bash
python convert_ckpt_to_safetensors.py --folder "/Users/username/Library/Containers/com.liuliu.draw-things/Data/Documents/Models"
```

#### Convert a Single File

Converts a specific file, saving the `.safetensors` in the same location:

```bash
python convert_ckpt_to_safetensors.py --file "c:/models/my_lora.ckpt"
```

### Command-Line Options

- **`--folder <path>`**: Convert all `.ckpt` files in folder (includes subfolders)
- **`--file <path>`**: Convert a single `.ckpt` file
- **`--inspect <path>`**: Show tensor names and metadata of an existing `.safetensors` file
- **`--overwrite`**: Overwrite existing `.safetensors` files (optional)
- **`--verbose`**: Print every tensor name during conversion (optional)
- **`--remove-ckpt`**: Disabled. Accepted for compatibility, does nothing. Originals are never deleted.

**Note:** You must use either `--folder` OR `--file`, not both.

### Examples

**Convert all files in a folder:**
```bash
python convert_ckpt_to_safetensors.py --folder "c:/models"
```

**Convert a single file:**
```bash
python convert_ckpt_to_safetensors.py --file "c:/models/sdxl_lora.ckpt"
```

**Convert with overwrite:**
```bash
python convert_ckpt_to_safetensors.py --folder "c:/models" --overwrite
```

**Get help:**
```bash
python convert_ckpt_to_safetensors.py --help
```

## Draw Things Model Locations

Default Draw Things model paths on macOS:

- **LoRAs**: `~/Library/Containers/com.liuliu.draw-things/Data/Documents/Models/`
- **Checkpoints**: Same directory as above

Use these paths with `--folder`.

## Output

The converter will:
1. Scan for `.ckpt` files in the input folder
2. Extract tensor data from each SQLite database
3. Convert tensors to appropriate PyTorch dtypes (float32, float16, etc.)
4. Save as `.safetensors` files in the output folder
5. Display a summary of successful, skipped, and failed conversions

Example output:
```
Draw Things .ckpt to .safetensors Converter
============================================================
Options:
  Overwrite existing: False
  Verbose mode: False

============================================================
Converting: models/my_lora.ckpt
============================================================
Found 2 tensors
  Skipping '__unet__[t-2-0]__': encoded with q8p. This tool cannot decode it.
  ⚠ Warning: 1 tensor(s) are quantized/compressed and were skipped.

Saving to models/my_lora.safetensors...
⚠ Partial conversion: saved 1 of 2 tensors (0.03 MB)
  The .safetensors file is NOT a faithful copy of the .ckpt. Keep the original.

============================================================
CONVERSION SUMMARY
============================================================
Total files: 1
✓ Successfully converted: 0
⚠ Partial (tensors missing, keep originals): 1
⊘ Skipped (already exists): 0
✗ Failed: 0
```

## Troubleshooting

### "No .ckpt files found"
- Check that your `--folder` path is correct
- Ensure the files have `.ckpt` extension
- Verify you have read permissions for the folder

### "Error converting: [Errno 2] No such file or directory"
- The input file path is incorrect
- Check the path passed to `--file` or `--folder`

### Import errors
- Make sure PyTorch and safetensors are installed: `pip install torch safetensors`
- Activate your virtual environment if using one

### Memory issues with large models
- The converter processes one tensor at a time to minimize memory usage
- If you still encounter issues, try converting files one at a time using `--file`

## Technical Details

Draw Things stores model weights in SQLite databases written by [s4nnc](https://github.com/liuliu/s4nnc) / [ccv](https://github.com/liuliu/ccv):
- **Table**: `tensors`
- **Columns**: `name`, `type`, `format`, `datatype`, `dim` (shape), `data` (tensor bytes)
- `type`: low 32 bits are the memory type, **high 32 bits are the codec identifier** of the `data` blob (0 = raw; `0x511` ezm7, `0xf7217` fpzip, `0x217` zip, `0x8a1e4b`..`0x8a1e8b` q4p..q8p, `0x8a1e9b`.. i8x). Bit `0x10000000` means the bytes live in `<file>-tensordata` and the blob is just two uint64: offset and length
- `datatype`: low 32 bits are the ccv datatype (`0x20000` F16, `0x04000` F32, `0x40000` palette-quantized)
- `dim`: 12 little-endian int32, trailing zeros unused
- Tensors written with the `externalData` codec keep their bytes in a sibling `<file>.ckpt-tensordata` file

The converter:
1. Reads the SQLite database
2. Skips any tensor whose codec identifier is non-zero or whose datatype is palette-quantized
3. Infers PyTorch dtype from bytes-per-element (so integer tensors come out as floats)
4. Reconstructs tensors with their shapes, keeping the internal names
5. Saves using the safetensors library

## Limitations

- Output keeps Draw Things' internal tensor names, so other tools will not recognise the file
- Compressed or quantized tensors (zip, ezm7, fpzip, q4p to q8p, i8x) and tensors kept in `-tensordata` files are skipped, not decoded
- dtype is guessed from byte counts, so integer tensors are written as floats
- Only works with Draw Things `.ckpt` files (SQLite format), not standard PyTorch `.ckpt` files
- Requires enough disk space for both input and output files

## Contributing

This repository is archived and does not accept contributions. Fork it if you want to continue the research; the mapping tables in [draw-things-community](https://github.com/drawthingsai/draw-things-community) are the place to start.

## License

MIT License - feel free to use and modify as needed.

## Acknowledgments

- Draw Things app by Liu Liu
- Safetensors library by Hugging Face
- PyTorch team

## Disclaimer

This tool is for converting your own legally obtained models. Ensure you have the right to use and convert any models before doing so. The authors are not responsible for any misuse of this tool.