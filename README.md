# EditableGAN
tensorflow implementation of "Editable GAN" ( 128 by 128 )

* Requirement
    * tensorflow 1.8
    * scipy
    * tensorpack for MNIST 
    
* Files
    * compare_ops
       * ops.py from https://github.com/google/compare_gan
    * mnist_main.py / mnist_model.py
       * experiment on mnist with rotation or color
    * main.py / model.py
       * experiment on celebA
       
* Run
    * python main.py --model_name hi --gpu 1 \[args ...\]
