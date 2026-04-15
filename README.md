# Not All Modalities at Once: Dynamic Dropout and Bidirectional Fusion for Robust Multi-modal Knowledge Graph Completion


## Details
- Python==3.9
- numpy==1.24.2
- scikit_learn==1.2.2
- torch==2.0.0
- tqdm==4.64.1
- transformers==4.28.0


## Data Preparation
You should first get the textual token embedding by running `save_token_embeddings.py` with transformers library (BERT, RoBERTa, LlaMA). You can first try MDBGF on the pre-processed datasets DB15K, MKG-W, and MKG-Y. The large token files in `tokens/` should be unzipped before using in the training process. We provide BEiT tokens for visual modality and BERT / RoBERTa / LlaMA tokens for textual modality.

## Train 

```bash
CUDA_VISIBLE_DEVICES=0 nohup python train_mygo_fgc.py --data MKG-W --num_epoch 5000 --hidden_dim 1024 --lr 6e-4 --dim 256 --max_vis_token 16 --max_txt_token 24 --num_head 4 --emb_dropout 0.9 --vis_dropout 0.4 --txt_dropout 0.1 --num_layer_dec 2 --mu 0.01 > log_MKG-W.txt &
```

More training scripts can be found in `run.sh`.

