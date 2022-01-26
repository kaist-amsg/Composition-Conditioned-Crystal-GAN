# Composition-Conditioned-Crystal-GAN
Composition-Conditioned Crystal GAN pytorch code

python3.6

pytorch 1.0.0

numpy 1.19.5

ase 3.22.0

tqdm 4.62.3



Because of the data capacity, training data must be created in "preparing_dataset" folder.

In "preparing_dataset" folder, please run "5.make_comp_dict.py" ~ "7.make_label.py" for making data-augmented mgmno dataset.

After the training dataset is created, run "train.py" by loading the dataset (opt.trainingdata in train.py).

If you want to download the dataset augmented to 2000 data per composition (i.e. "mgmno_2000.pickle"), follow this link-> https://figshare.com/s/0dce6bb830ae1e392206
