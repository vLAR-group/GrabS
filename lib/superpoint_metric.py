import torch
import numpy as np

def correctness(sp, gt, gt_classes):
    ### -1 is invalid index
    valid_mask = (sp!=-1) & (gt!=-1)
    sp, gt = sp[valid_mask], gt[valid_mask]
    out_label = -torch.ones_like(gt)
    for i in torch.unique(sp):
        major = torch.mode(gt[sp==i]).values
        out_label[sp==i] = major

    '''Unsupervised, Match pred to gt'''
    sem_num = gt_classes
    mask = (out_label >= 0) & (out_label < sem_num)
    histogram = np.bincount(sem_num * gt[mask] + out_label[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    o_Acc = histogram[range(sem_num), range(sem_num)].sum() / histogram.sum()*100.
    m_Acc = np.mean(histogram[range(sem_num), range(sem_num)] / histogram.sum(1))*100

    '''Final Metrics'''
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)
    F1_score = 2*tp/ (2*tp + fp + fn +1e-8)

    return o_Acc, m_Acc, s, F1_score


def BR(sp, gt, gt_classes):
    ### Boundary Recall
    ### -1 is invalid index
    valid_mask = (sp != -1) & (gt != -1)
    sp, gt = sp[valid_mask], gt[valid_mask]
    out_label = -torch.ones_like(gt)
    for i in torch.unique(sp):
        major = torch.mode(gt[sp == i]).values
        out_label[sp == i] = major

    '''Unsupervised, Match pred to gt'''
    sem_num = gt_classes
    mask = (out_label >= 0) & (out_label < sem_num)
    histogram = np.bincount(sem_num * gt[mask] + out_label[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    o_Acc = histogram[range(sem_num), range(sem_num)].sum() / histogram.sum() * 100.
    m_Acc = np.mean(histogram[range(sem_num), range(sem_num)] / histogram.sum(1)) * 100

    '''Final Metrics'''
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)

    return o_Acc, m_Acc, s


def BP(sp, gt, gt_classes):
    ### Boundary Precision
    ### -1 is invalid index
    valid_mask = (sp != -1) & (gt != -1)
    sp, gt = sp[valid_mask], gt[valid_mask]
    out_label = -torch.ones_like(gt)
    for i in torch.unique(sp):
        major = torch.mode(gt[sp == i]).values
        out_label[sp == i] = major

    '''Unsupervised, Match pred to gt'''
    sem_num = gt_classes
    mask = (out_label >= 0) & (out_label < sem_num)
    histogram = np.bincount(sem_num * gt[mask] + out_label[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    o_Acc = histogram[range(sem_num), range(sem_num)].sum() / histogram.sum() * 100.
    m_Acc = np.mean(histogram[range(sem_num), range(sem_num)] / histogram.sum(1)) * 100

    '''Final Metrics'''
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)

    return o_Acc, m_Acc, s


def under_seg_error(sp, gt, gt_classes):
    ### -1 is invalid index
    valid_mask = (sp != -1) & (gt != -1)
    sp, gt = sp[valid_mask], gt[valid_mask]
    out_label = -torch.ones_like(gt)
    for i in torch.unique(sp):
        major = torch.mode(gt[sp == i]).values
        out_label[sp == i] = major

    '''Unsupervised, Match pred to gt'''
    sem_num = gt_classes
    mask = (out_label >= 0) & (out_label < sem_num)
    histogram = np.bincount(sem_num * gt[mask] + out_label[mask], minlength=sem_num ** 2).reshape(sem_num, sem_num)
    o_Acc = histogram[range(sem_num), range(sem_num)].sum() / histogram.sum() * 100.
    m_Acc = np.mean(histogram[range(sem_num), range(sem_num)] / histogram.sum(1)) * 100

    '''Final Metrics'''
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = '| mIoU {:5.2f} | '.format(100 * m_IoU)
    for IoU in IoUs:
        s += '{:5.2f} '.format(100 * IoU)

    return o_Acc, m_Acc, s