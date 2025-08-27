
# RSD

apt update

apt install git-lfs

git clone https://github.com/sudokill/RSD/

python3 -m venv venv

source venv/bin/activate

pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

python -c "import torch; print(torch.cuda.is_available())"

pip install -r requirements.txt

git clone https://huggingface.co/datasets/Yzl-code/RS-Diffusion

--MULTI GPU--

accelerate config
accelerate launch train_RS_real.py --config config/train_official_data_config.yaml

--SINGLE GPU--

python train_RS_real.py --config config/train_official_data_config.yaml


