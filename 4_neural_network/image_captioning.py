import time, os, json
import numpy as np
import matplotlib.pyplot as plt

import coco_utils
from captioning_solver import CaptioningSolver
from image_utils import image_from_url
from rnn import CaptioningRNN


def main():
    # The dataset (987M) can be downloaded from
    # https://drive.google.com/file/d/1Wgeq3NZ4R1letnZEKLo-DTSSgcTsgkmq/view?usp=sharing
    # The dataset contains the feature of images in MSCOCO dataset
    # The data should be in the same folder as the code
    # Load COCO data from disk; this returns a dictionary
    small_data = coco_utils.load_coco_data(max_train=50)

    # Experiment with vanilla RNN
    small_rnn_model = CaptioningRNN(
          cell_type='rnn',
          word_to_idx=small_data['word_to_idx'],
          input_dim=small_data['train_features'].shape[1],
          hidden_dim=512,
          wordvec_dim=256,
    )

    small_rnn_solver = CaptioningSolver(small_rnn_model, small_data,
           update_rule='adam',
           num_epochs=58,
           batch_size=25,
           optim_config={
             'learning_rate': 4e-3,
           },
           lr_decay=0.95,
           verbose=True, print_every=10,
         )

    small_rnn_solver.train()

    # Plot the training losses
    plt.plot(small_rnn_solver.loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training loss history')
    plt.savefig('train_loss_ic.png', dpi=250)
    plt.close()

    for split in ['train', 'val']:
        # some images might be deprecated. You may rerun the code several times
        # to successfully get the sample images from url.
        minibatch = coco_utils.sample_coco_minibatch(
            small_data, split=split, batch_size=2, seed=0)
        gt_captions, features, urls = minibatch
        gt_captions = coco_utils.decode_captions(gt_captions,
                                                 small_data['idx_to_word'])

        sample_captions = small_rnn_model.sample(features)
        sample_captions = coco_utils.decode_captions(sample_captions,
                                                     small_data['idx_to_word'])

        for i, (gt_caption, sample_caption, url) in enumerate(zip(gt_captions, sample_captions, urls)):
            plt.imshow(image_from_url(url))
            plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
            plt.axis('off')
            plt.savefig('sample_caption_ic_'+ str(i) +'.png', dpi=250)
            plt.close()


if __name__== "__main__":
    main()
