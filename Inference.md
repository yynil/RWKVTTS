# Install the code base and the dependencies
```bash
git clone https://github.com/yynil/RWKVTTS
git clone https://github.com/yynil/CosyVoice
```

# Install the dependencies
```bash
conda create -n rwkvtts-311 -y python=3.11
conda activate rwkvtts-311
conda install -y -c conda-forge pynini==2.1.6
cd RWKVTTS
pip install -r rwkvtts_requirements.txt
``` 

