# seq len 2048
export TPU_NAME='v4-256'
export ZONE='us-central2-b'

echo "[local] Killing TPU"
gcloud compute tpus tpu-vm ssh beomi@$TPU_NAME \
    --zone $ZONE --worker=all --command "sudo fuser -k /dev/accel0"

echo "[local] Removing TPU Lock"
gcloud compute tpus tpu-vm ssh beomi@$TPU_NAME \
    --zone $ZONE --worker=all --command "sudo rm -f /tmp/libtpu_lockfile"

echo "[local] Removing screens"
gcloud compute tpus tpu-vm ssh beomi@$TPU_NAME \
    --zone $ZONE --worker=all --command "killall screen"

echo "[local] Git pull"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command \
    "source EasyLMenv2/bin/activate && \
cd EasyLM-o && git pull && rm /home/beomi/EasyLM-o/runner.sh"

echo "[local] Set runner.sh"

HASH='2a4ab2be67e24a559ec6d780dc786c14'

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone $ZONE --worker=all --command "
cat > /home/beomi/EasyLM-o/runner.sh << 'EOF'
export LIBTPU_INIT_ARGS='--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --xla_tpu_enable_latency_hiding_scheduler=true TPU_MEGACORE=MEGACORE_DENSE'
export NAME=7B
python -m EasyLM.models.llama.llama_train \
--load_dataset_state=gs://jaxseq-test/easylm-out/llama-2-ko-7b-kor_shuffle_dataset/$HASH/dataset.pkl \
--load_checkpoint=trainstate::gs://jaxseq-test/easylm-out/llama-2-ko-7b-kor_shuffle_dataset/$HASH/streaming_train_state \
--mesh_dim=4,-1,1 \
--dtype=bf16 \
--total_steps=100001 \
--log_freq=50 \
--save_model_freq=2500 \
--save_milestone_freq=10000 \
--load_llama_config=7b \
--update_llama_config='{\"attn_pdrop\": 0.1}' \
--train_dataset.type='json' \
--train_dataset.text_processor.fields='text' \
--train_dataset.json_dataset.path=gs://kodataset/20231208_modu_aihub_kowiki_all_shuffle.jsonl \
--train_dataset.json_dataset.seq_length=2048 \
--train_dataset.json_dataset.batch_size=1024 \
--tokenizer.name=beomi/llama-2-ko-7b \
--tokenizer.bos_token='<s>' \
--tokenizer.eos_token='</s>' \
--tokenizer.pad_token='</s>' \
--optimizer.type=adamw \
--optimizer.adamw_optimizer.weight_decay=0.1 \
--optimizer.adamw_optimizer.lr=5e-5 \
--optimizer.adamw_optimizer.end_lr=1e-5 \
--optimizer.adamw_optimizer.lr_warmup_steps=2000 \
--optimizer.adamw_optimizer.lr_decay_steps=100001 \
--checkpointer.save_optimizer_state=True \
--checkpointer.float_dtype=bf16 \
--logger.online=True \
--logger.output_dir=gs://jaxseq-test/easylm-out/llama-2-ko-7b-kor_shuffle_dataset
EOF
chmod +x /home/beomi/EasyLM-o/runner.sh"

echo "[local] RUN!!!"

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone us-central2-b --worker=all --command \
    "screen -L -d -m bash -i -c 'export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=107374182400; \
source EasyLMenv2/bin/activate && cd EasyLM-o; /home/beomi/EasyLM-o/runner.sh'"

# gcloud compute tpus tpu-vm ssh $TPU_NAME --zone us-central2-b --worker=all --command \
# "export TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD=107374182400; \
# source EasyLMenv2/bin/activate && cd EasyLM-o; /home/beomi/EasyLM-o/runner.sh"
