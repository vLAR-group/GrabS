import os
import numpy as np

# other_cls = ['02828884', '02933112', '03211117', '03636649', '04256520', '04379243']
other_cls = ['02691156', '02828884', '03211117', '03636649', '03691459', '04401088', '02933112', '04379243', '04530566', '04090263', '03001627', '04256520']
data_path = './GAPS_SDF'
# out_path = './data/other_cls_data'
# split_path = './data/shapenet_splits/data'

out_path = './data/other_cls_data'
split_path = './data/shapenet_splits/data'

for cate in other_cls:
    splits = os.path.join(split_path, cate, 'train.lst')
    with open(splits, 'r') as f:
        files = f.read().split('\n')
        files = list(filter(lambda x: len(x) > 0, files))

    for file in files:
        pc_file = os.path.join(data_path, cate, file,  'pointcloud.npz')
        points = np.load(pc_file)['points'].astype(np.float32)
        points = points[np.random.choice(len(points), 10000, replace=False)]
        os.makedirs(os.path.join(out_path, cate, file), exist_ok=True)
        np.save(os.path.join(out_path, cate, file, 'pointcloud.npy'), points)
        print('save to', os.path.join(out_path, cate, file, 'pointcloud.npy'))
