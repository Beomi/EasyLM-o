# seq len 2048
export TPU_NAME='v4-256'
export ZONE='us-central2-b'

echo "[local] Killing TPU"
gcloud compute tpus tpu-vm ssh $TPU_NAME \
--zone $ZONE --worker=all --command "sudo fuser -k /dev/accel0; sudo rm /tmp/libtpu_lockfile"
