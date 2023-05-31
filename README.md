# VisionTransformer
***

NOTE: *THIS IS A MINIATURE VERSION RESULTS ARE NOT IDENTICAL OR EVEN CLOSE TO THAT OF THE PAPER.*

This repository is a **miniature** implementation of the paper [Restormer: Efficient Transformer for High-Resolution Image Restoration](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf).

Things that I have implemented:
* Made the architecture shallow only one level deep.
* Focused more on the transformer blocks [MDTA](./transfomer/mdta.py) and [GDFN](./transfomer/gdfn.py).
* I have explained the transfomer blocks in comments and will also soon enough post a blog on medium.
  
Things that I have not implemented:
* Progressive learning.

***

## To run the code:

1. Clone the repository.
2. Download the raindrop dataset from [here](https://drive.google.com/drive/folders/1e7R76s6vwUJxILOcAsthgDLPSnOrQ49K) and paste in the cloned repo.
3. Rename the dataset directory such that all the files are present in `.\data\RainDrop\train\` similar to the following structure.

```
VisionTransformer
|--main.py
|--data
   |--RainDrop
      |--train
         |--data
         |--GT

```

Execute following comands in the given order:

`cd VisionTransformer/` \
`python main.py`