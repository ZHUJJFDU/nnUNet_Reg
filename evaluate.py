import os
import numpy as np
import nibabel as nib


def compute(mask1, mask2, num_classes):
    dice_scores = []
    iou_scores = []
    recall_scores = []

    for class_id in range(1, num_classes + 1):
        mask1_class = (mask1 == class_id).astype(np.uint8)
        mask2_class = (mask2 == class_id).astype(np.uint8)

        # Dice
        intersection = np.sum(mask1_class * mask2_class)
        denominator = np.sum(mask1_class) + np.sum(mask2_class)
        if denominator == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2.0 * intersection) / denominator
        dice_scores.append(dice)

        # IoU
        union = np.sum(np.logical_or(mask1_class, mask2_class))
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = intersection / union
        iou_scores.append(iou)

        # Recall
        tp = intersection
        fn = np.sum(mask1_class) - tp
        if mask1_class.sum() == 0:
            if mask2_class.sum() == 0:
                recall = 1.0
            else:
                recall = 0.0
        else:
            recall = tp / (tp + fn + 1e-8)
        recall_scores.append(recall)

    return dice_scores, iou_scores, recall_scores


def main(folder1, folder2, num_classes=3):
    file_names = [f for f in os.listdir(folder1) if f.endswith('.nii.gz')]

    total_dice = np.zeros(num_classes)
    total_iou = np.zeros(num_classes)
    total_recall = np.zeros(num_classes)
    count = 0

    results = {}

    for file_name in file_names:
        path1 = os.path.join(folder1, file_name)
        path2 = os.path.join(folder2, file_name)

        if not os.path.exists(path1) or not os.path.exists(path2):
            print(f"Warning: File {file_name} not found in one of the folders.")
            continue

        try:
            img1 = nib.load(path1)
            img2 = nib.load(path2)
        except Exception as e:
            print(f"Error loading file {file_name}: {e}")
            continue

        data1 = img1.get_fdata()
        data2 = img2.get_fdata()

        if data1.shape != data2.shape:
            print(f"Warning: Shape mismatch for file {file_name}. Skipping.")
            continue

        dice, iou, recall = compute(data1, data2, num_classes)

        results[file_name] = {'Dice': dice, 'IoU': iou, 'Recall': recall}

        total_dice += np.array(dice)
        total_iou += np.array(iou)
        total_recall += np.array(recall)
        count += 1

        print(f"File: {file_name}")
        for class_id in range(1, num_classes + 1):
            print(f"  Class {class_id} - Dice: {dice[class_id - 1]:.4f}, IoU: {iou[class_id - 1]:.4f}, Recall: {recall[class_id - 1]:.4f}")
        print("-" * 50)

    if count > 0:
        avg_dice = total_dice / count
        avg_iou = total_iou / count
        avg_recall = total_recall / count

        print("\nOverall Statistics:")
        for class_id in range(1, num_classes + 1):
            print(
                f"  Class {class_id} - Average Dice: {avg_dice[class_id - 1]:.4f}, "
                f"Average IoU: {avg_iou[class_id - 1]:.4f}, "
                f"Average Recall: {avg_recall[class_id - 1]:.4f}"
            )
    else:
        print("No valid files found.")

    return results, avg_dice, avg_iou, avg_recall

if __name__ == "__main__":
    folder1 = r'C:\Users\Administrator\Desktop\nnUNet-master\DATASET\nnUNet_raw\Dataset103_new\labelsTs'
    folder2 = r'C:\Users\Administrator\Desktop\2\after'
    num_classes = 3  # 根据你的标签类别数量设置

    results, avg_dice, avg_iou, avg_recall = main(folder1, folder2, num_classes)