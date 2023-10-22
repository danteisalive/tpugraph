  
## Training-CPU 


```bash
conda create --name llm4tpu-cpu python=3.11
conda activate llm4tpu-cpu
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch_geometric
pip install pytorch_lightning
pip install yacs
pip install matplotlib numpy pandas 


python main_layout.py --cfg configs/tpugraphslayout.yaml 
```

## Training-GPU 