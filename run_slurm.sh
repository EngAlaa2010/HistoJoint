#!/bin/bash
#SBATCH --account PAS3015
#SBATCH --job-name RNA_PROTEIN
#SBATCH --time=2:30:00 # Requesting for 1 min
#SBATCH --ntasks-per-node=1 # Number of cores on a single node or number of tasks per requested node. Default is a single core.
#SBATCH --nodes=1 # Number of nodes.
#SBATCH --gpus-per-node=1 # Number of gpus per node. Default is none.
#SBATCH --cpus-per-task=32             # Number of CPU cores per task
#SBATCH --mem=32gb # Specify the (RAM) main memory required per node. # to request 24gb use --mem=24gb or --mem=24000mb (Other flags which are mutually exclusive: >> 1st flag. --mem-per-gpu=0 # real memory required per allocated GPU # usage: can be 0 (all memory), 40G, 80G. >> 2nd flag. --mem-per-cpu=0 # MB # maximum amount of real memory per allocated cpu required by the job. --mem >= --mem-per-cpu if --mem is specified. # usage e.g.: type `4G` for 4 gigbytes)
#SBATCH --output=./slurmoutput/train_model_%j.out #Standard output log
#SBATCH --error=./slurmoutput/train_model_%j.err #Standard output log
#SBATCH --cluster=ascend # Can also explicitly specify which cluster to submit the job to. Or, log in to the node and submit the job.


# can add these lines here if you haven't called them first in the terminal
# load miniconda (see command/miniconda version in the terminal first) if not loaded in the terminal before
module load cuda/11.8.0
module load miniconda3/24.1.2-py310
conda activate multi310

# run the python file
python -c "import torch; print(torch.__version__); print('cuda available:', torch.cuda.is_available()); print(torch.version.cuda)"

cd /fs/scratch/PAS3015/alaa/HE_RNA_Protein
#python scripts/03_encode_patches_dinov2.py \
#  --base /fs/scratch/PAS3015/alaa/HE_RNA_Protein \
 # --patch-index /fs/scratch/PAS3015/alaa/HE_RNA_Protein/outputs/patch_index.csv \
 # --outdir /fs/scratch/PAS3015/alaa/HE_RNA_Protein/outputs_hires \
 # --dinov2-arch dinov2_vitb14 \
 # --proj-dim 256 \
 # --image-size 224 \
 # --normalize imagenet \
 # --batch-size 64 \
 # --workers 4 \
 # --device auto \
 # --save-concat \
 # --overwrite \
 # -v


#step 6
#python scripts/06_train_rna_head.py \
#  --base /fs/scratch/PAS3015/alaa/HE_RNA_Protein \
#  --standardize-y \
#  --device auto \
#  -v


#python scripts/06_train_rna_mlp_head.py \
#  --base /fs/scratch/PAS3015/alaa/HE_RNA_Protein \
 # --outdir /fs/scratch/PAS3015/alaa/HE_RNA_Protein/outputs/rna_mlp_head \
 # --standardize-y \
  #--device auto \
  #-v


#python scripts/07_train_protein_head.py -v --only-in-tissue --standardize-y --device auto

#step 7 
#Predict ALL proteins
#python scripts/07_train_protein_head.py -v --only-in-tissue --standardize-y --device auto --exclude-isotypes

#Predict top-6 highest variance proteins
#python scripts/07_train_protein_head.py -v --only-in-tissue --standardize-y --device auto --exclude-isotypes --topk-variance 6


#Predict custom list of protein 
#python scripts/07_train_protein_head.py -v --only-in-tissue --standardize-y --device auto --exclude-isotypes --proteins KRT5-1 EPCAM-1 VIM-1 CD3E-1 CD8A-1 HLA-DRA


#step 8 
#learn RNA and Protein only 6 protein

#python scripts/08_train_joint_latent.py -v --device cuda --standardize-inputs \
 # --prot-subset outputs/prot_feature_names.csv --epochs 200 --patience 20
#python scripts/08_train_joint_latent.py \
  #--base /fs/scratch/PAS3015/alaa/HE_RNA_Protein \
  #--prot-subset /fs/scratch/PAS3015/alaa/HE_RNA_Protein/outputs/prot_feature_names.csv \
  #--standardize-inputs \
  #--latent-dim 64 \
  #--enc-hidden 256 256 \
  #--dropout 0.1 \
  #--batch-size 256 \
  #--epochs 200 \
  #--patience 20 \
  #--device auto \
  #-v


#step 9 : 
#learn from morphology to RNA first: 
#python scripts/09_train_morph_to_joint.py -v --only-in-tissue --device auto

#predict the RNA and Protein from Morphology:
#python scripts/09_train_morph_to_joint.py -v --only-in-tissue --device auto --predict-u --lambda-u 0.5


#run (balanced: match BOTH RNA-latent and Protein-latent)
#python scripts/09_train_morph_to_joint.py -v --only-in-tissue --device auto --overwrite

#Protein-anchored retrieval (bias training toward protein latent)
#python scripts/09_train_morph_to_joint.py -v --only-in-tissue --device cuda --overwrite \
#  --lambda-z 0.5 --lambda-u 1.5

#Add MSE stabilizer (contrastive + “pull to exact target”)
#python scripts/09_train_morph_to_joint.py -v --only-in-tissue --device cuda --overwrite \
#  --lambda-z 1.0 --lambda-u 1.0 --lambda-mse-z 0.05 --lambda-mse-u 0.05



#python scripts/09_train_morph_to_joint.py -v --only-in-tissue --device cuda --overwrite --early-stop-metric u_top1


#python scripts/09_train_morph_to_joint.py \
 # --base /fs/scratch/PAS3015/alaa/HE_RNA_Protein \
  #--only-in-tissue \
  #--device cuda \
  #--batch-size 256 \
  #-v \
  #--overwrite

#python scripts/09_train_morph_to_joint.py \
#  --base /fs/scratch/PAS3015/alaa/HE_RNA_Protein \
#  --device cuda -v --overwrite \
#  --batch-size 512 \
#  --tau 0.05 \
#  --lambda-mse-u 0.1 --lambda-mse-z 0.1

#python scripts/09_train_morph_to_joint.py \
#  --base /fs/scratch/PAS3015/alaa/HE_RNA_Protein \
#  --device cuda -v --overwrite \
#  --batch-size 512 \
#  --tau 0.06 \
#  --lambda-z 0.0 --lambda-u 1.0 \
#  --lambda-mse-u 0.1


BASE=/fs/scratch/PAS3015/alaa/HE_RNA_Protein
OUTBASE=$BASE/outputs/morph_to_joint_runs

for SEED in 0 1 2 3 4; do
  python scripts/09_train_morph_to_joint.py \
    --base $BASE \
    --outdir $OUTBASE/seed_${SEED}_tile300_tau005_b512_lz1_lu1_mse005 \
    --device cuda -v --overwrite \
    --seed $SEED \
    --tile-size 300 \
    --batch-size 512 \
    --tau 0.05 \
    --lambda-z 1.0 --lambda-u 1.0 \
    --lambda-mse-z 0.05 --lambda-mse-u 0.05
done
