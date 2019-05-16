# EditableGAN
tensorflow implementation of "Editable GAN" ( 128 by 128 )

* Requirement
    * tensorflow 1.8
    * scipy
    * tensorpack for MNIST 
    * tqdm
    * python 3.6
    
* Files
    * compare_ops
       * ops.py from https://github.com/google/compare_gan
    * mnist_main.py / mnist_model.py
       * experiment on mnist with rotation or color
    * main.py / model.py
       * experiment on celebA
    * data_loader.py
      * Dataset implementation for "celebA".
      * Read image and label - label can be determined ( change "atts" in [a-z]{2-4}gan_celebA not "att_dict")
      * three modes - val : 10,000 / test : 10,000 / train : others
   * utils.py
      * Just utils ...
   * ops.py
      * Just ops ...
      
* DATASET ( for celebA )
   * for example,
   ```
     data
      |--- img_blah_blah ( ex img_celeba_128_center )
      |--- img_align_celeba
                  | ---- 000001.png ( or jpg.. change the code )
                  | ---- 000002.png ...
      |--- list_attr_celeba.txt
     GANs
      |--- sndcgan_celebA.py
   ```
   * Then, execute : python main.py --gpu 1 --img_dir ../data/img_align_celeba --txt_dir ../data/list_attr_celeba.txt
