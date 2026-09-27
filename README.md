# RMSNorm

A simple PyTorch RMSNorm reference with space for a CUDA implementation.

## Project structure

```text
├── rmsnorm.py          # PyTorch reference and basic verification
├── src/
│   ├── rmsnorm.cu      # CUDA implementation placeholder
│   └── bindings.cpp    # C++ bindings placeholder
├── benchmarks/
├── tests/
├── results/
└── README.md
```

## Run the reference

With PyTorch installed in your Python environment:

```sh
python rmsnorm.py
```

The script checks the output values and shape.
