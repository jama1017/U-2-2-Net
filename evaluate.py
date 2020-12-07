import numpy as np
import argparse
import os
import random
from PIL import Image
import glob

import pathlib
import matplotlib.pyplot as plt

output_mask_dir = pathlib.Path('DUTS-TE-our-gcp-all')
gt_mask_dir = pathlib.Path('DUTS-TE-Mask')
u2_mask_dir = pathlib.Path('DUTS-TE-Paper')

def evaluation():
    """Given a predicted saliency probability map, its precision and recall scores are computed by comparing its 
    thresholded binary mask against the ground truth mask. The precision and recall of a dataset are computed by averaging the
    precision and recall scores of those saliency maps. By varying the thresholds from 0 to 1, we can obtain a set of average
    precision-recall pairs of the dataset."""
    # Precision = True Positives / (True Positives + False Positives)
    # Recall = True Positives / (True Positives + False Negatives)
    output_masks = []
    output_m = sorted(glob.glob(str(output_mask_dir.joinpath("*.jpg"))))
    if len(output_m) == 0:
        print("No image found in our output mask directory: {}".format(output_mask_dir))
    else:
        output_masks.extend(output_m)
    
    gt_masks = []
    gt_m = sorted(glob.glob(str(gt_mask_dir.joinpath("*.png"))))
    if len(gt_m) == 0:
        print("No image found in ground truth mask directory: {}".format(gt_mask_dir))
    else:
        gt_masks.extend(gt_m)

    u2_output_masks = []
    u2_m = sorted(glob.glob(str(u2_mask_dir.joinpath("*.png"))))
    if len(u2_m) == 0:
        print("No image found in u2net paper mask directory: {}".format(u2_mask_dir))
    else:
        u2_output_masks.extend(u2_m)
    
    # Precision-Recall and F measure
    thresholds = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    precisions = []
    recalls = []
    precisions_u2 = []
    recalls_u2 = []
    num_images = len(gt_masks)
    assert num_images == len(u2_output_masks)
    assert num_images == len(output_masks)

    for thres in thresholds:
        cur_precision = 0
        cur_recall = 0
        cur_precision_2 = 0
        cur_recall_2 = 0
        invalid_images_ours = 0
        invalid_images_u2 = 0
        print(thres)
        for i in range(num_images):
            output_mask = np.array(Image.open(output_masks[i]).convert("L"))/255
            paper_output_mask = np.array(Image.open(u2_output_masks[i]).convert("L"))/255
            gt_mask = np.array(Image.open(gt_masks[i]).convert("L"))/255
            assert output_mask.shape == gt_mask.shape
            assert paper_output_mask.shape == gt_mask.shape
            
            binary_output_mask = np.where(output_mask >= thres, 1, 0)
            binary_output_mask_p = np.where(paper_output_mask >= thres, 1, 0)
            comparison_mask = gt_mask - binary_output_mask
            sum_mask = gt_mask + binary_output_mask
            comparison_mask_2 = gt_mask - binary_output_mask_p
            sum_mask_2 = gt_mask + binary_output_mask_p
            tp = np.count_nonzero(sum_mask==2)
            fp = np.count_nonzero(comparison_mask==-1)
            fn = np.count_nonzero(comparison_mask==1)
            tp2 = np.count_nonzero(sum_mask_2==2)
            fp2 = np.count_nonzero(comparison_mask_2==-1)
            fn2 = np.count_nonzero(comparison_mask_2==1)
            try:
                precision = tp/(tp+fp)
                recall = tp/(tp+fn)
            except ZeroDivisionError:
                invalid_images_ours += 1
            cur_precision += precision
            cur_recall += recall
            
            try:
                precision_2 = tp2/(tp2+fp2)
                recall_2 = tp2/(tp2+fn2)
            except ZeroDivisionError:
                invalid_images_u2 += 1
            
            cur_precision_2 += precision_2
            cur_recall_2 += recall_2
            
        precisions.append(cur_precision/(num_images-invalid_images_ours))
        recalls.append(cur_recall/(num_images-invalid_images_ours))
        precisions_u2.append(cur_precision_2/(num_images-invalid_images_u2))
        recalls_u2.append(cur_recall_2/(num_images-invalid_images_u2))

    plt.plot(recalls, precisions, 'b', label='Our Model')
    plt.plot(recalls_u2, precisions_u2, 'r', label="U2Net Paper")
    plt.legend(loc="lower left")
    plt.axis([0.5,1,0.5,1])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    precisions_u2 = np.array(precisions_u2)
    recalls_u2 = np.array(recalls_u2)
    maxF_u2 = np.amax(np.multiply(precisions_u2, recalls_u2) * 1.09/(precisions_u2 * 0.09+recalls_u2))
    maxF_ours = np.amax(np.multiply(precisions, recalls) * 1.09/(precisions * 0.09+recalls))
    print("Max F measure for original paper:", maxF_u2)
    print("Max F measure for our model:", maxF_ours)

    # MAE
    ours_error = 0
    u2_error = 0
    for i in range(num_images):
        output_mask = np.array(Image.open(output_masks[i]).convert("L"))/255
        paper_output_mask = np.array(Image.open(u2_output_masks[i]).convert("L"))/255
        gt_mask = np.array(Image.open(gt_masks[i]).convert("L"))/255
        assert output_mask.shape == gt_mask.shape
        assert paper_output_mask.shape == gt_mask.shape
        cur_error = np.mean(np.absolute(gt_mask - output_mask))
        cur_u2_error = np.mean(np.absolute(gt_mask - paper_output_mask))
        ours_error += cur_error
        u2_error += cur_u2_error
    print("MAE for original paper:", u2_error/num_images)
    print("MAE for our model:", ours_error/num_images)


evaluation()