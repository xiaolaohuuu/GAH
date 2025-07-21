# GAH
The code of Generative Augmentation Hashing for Few-shot Cross-Modal Retrieval
## Play with GAH
Before running the main training script, you must first generate the necessary augmented data. This step prepares the synthetic image-text pairs used for later training.
```bash
python Cycle_Generative_Augmentation.py
```
After the augmented data has been generated, you can start the training process:
```bash
python main.py 
```
## Dataset
The preprocessed version of the Flickr-25K (k=8) dataset used in our experiments is publicly available at:[Baidu Netdisk Download Link](https://pan.baidu.com/s/1BiQV2WHY7gYxdYoA-iJ08g) (Extraction Code: `fp7e`)