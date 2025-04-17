import re
from pathlib import Path
import numpy as np
import pandas as pd
from fire import Fire
from natsort import natsorted
from loguru import logger
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from base_preprocessing import BasePreprocessing
from point_cloud_utils import load_ply_with_normals, write_point_cloud_in_ply
from typing import List, Tuple
import colorsys
import matplotlib.pyplot as plt

from scannet200_constants import (
    SCANNET_COLOR_MAP_20,
    VALID_CLASS_IDS_20)

def get_evenly_distributed_colors(count: int) -> List[Tuple[np.uint8, np.uint8, np.uint8]]:
    HSV_tuples = [(x / count, 1.0, 1.0) for x in range(count)]
    return list(map(lambda x: (np.array(colorsys.hsv_to_rgb(*x)) * 255).astype(np.uint8),HSV_tuples))
colormap = []
for _ in range(1000):
    for k in range(12):
        colormap.append(plt.cm.Set3(k))
    for k in range(9):
        colormap.append(plt.cm.Set1(k))
    for k in range(8):
        colormap.append(plt.cm.Set2(k))
colormap.append((0, 0, 0, 0))
colormap = np.array(colormap)[:, 0:3]*255

class ScannetPreprocessing(BasePreprocessing):
    def __init__(
        self,
        data_dir: str = "./data/scannet/raw",
        save_dir: str = "./data/scannet/processed",
        modes: tuple = ("train", "validation", "test"),
        n_jobs: int = -1,
        vis: bool = True,
        git_repo: str = "./ScanNet"):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        self.vis = vis
        git_repo = Path(git_repo)
        self.create_label_database(git_repo)
        for mode in self.modes:
            trainval_split_dir = git_repo / "Tasks" / "Benchmark"
            scannet_special_mode = "val" if mode == "validation" else mode
            with open(trainval_split_dir / (f"scannetv2_{scannet_special_mode}.txt")) as f:
                # -1 because the last one is always empty
                split_file = f.read().split("\n")[:-1]

            scans_folder = "scans_test" if mode == "test" else "scans"
            filepaths = []
            for scene in split_file:
                filepaths.append(self.data_dir/ scans_folder/ scene/ (scene + "_vh_clean_2.ply"))
            self.files[mode] = natsorted(filepaths)
        self.preprocess()

    def create_label_database(self, git_repo):
        if (self.save_dir / "label_database.yaml").exists():
            return self._load_yaml(self.save_dir / "label_database.yaml")
        df = pd.read_csv(self.data_dir / "scannetv2-labels.combined.tsv", sep="\t")
        df = (df[~df[["nyu40class", "nyu40id"]].duplicated()][["nyu40class", "nyu40id"]]
            .set_index("nyu40id").sort_index()[["nyu40class"]].rename(columns={"nyu40class": "name"})
            .replace(" ", "_", regex=True))
        # df = pd.DataFrame([{"name": "empty"}]).append(df)
        df = pd.concat([pd.DataFrame([{"name": "empty"}]), df])
        df["validation"] = False

        with open(git_repo/ "Tasks"/ "Benchmark"/ "classes_SemVoxLabel-nyu40id.txt") as f:
            for_validation = f.read().split("\n")
        for category in for_validation:
            index = int(re.split(" +", category)[0])
            df.loc[index, "validation"] = True

        # doing this hack because otherwise I will have to install imageio
        with open(git_repo / "BenchmarkScripts" / "util.py") as f:
            util = f.read()
            color_list = eval("[" + util.split("return [\n")[1])

        df["color"] = color_list

        label_database = df.to_dict("index")
        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    def process_file(self, filepath, mode):
        """process_file.

        Please note, that for obtaining segmentation labels ply files were used.

        Args:
            filepath: path to the main ply file
            mode: train, test or validation

        Returns:
            filebase: info about file
        """
        scene, sub_scene = self._parse_scene_subscene(filepath.name)
        filebase = {
            "filepath": filepath,
            "scene": scene,
            "sub_scene": sub_scene,
            "raw_filepath": str(filepath),
            "file_len": -1}
        # reading both files and checking that they are fitting
        coords, features, _ = load_ply_with_normals(filepath)
        file_len = len(coords)
        filebase["file_len"] = file_len
        ###################!!!
        points = np.hstack((coords, features))#features are RGB NxNyNz

        if mode in ["train", "validation"]:
            # getting scene information
            description_filepath = Path(filepath).parent / filepath.name.replace("_vh_clean_2.ply", ".txt")
            with open(description_filepath) as f:
                scene_type = f.read().split("\n")[:-1]
            scene_type = scene_type[-1].split(" = ")[1]
            filebase["scene_type"] = scene_type
            filebase["raw_description_filepath"] = description_filepath

            # getting instance info
            instance_info_filepath = next(Path(filepath).parent.glob("*.aggregation.json"))
            segment_indexes_filepath = next(Path(filepath).parent.glob("*[0-9].segs.json"))
            instance_db = self._read_json(instance_info_filepath)
            segments = self._read_json(segment_indexes_filepath)
            segments = np.array(segments["segIndices"])
            filebase["raw_instance_filepath"] = instance_info_filepath
            filebase["raw_segmentation_filepath"] = segment_indexes_filepath

            # add segment id as additional feature
            segment_ids = np.unique(segments, return_inverse=True)[1]
            ###########################################!!!
            points = np.hstack((points, segment_ids[..., None]))

            # reading labels file
            label_filepath = filepath.parent / filepath.name.replace(".ply", ".labels.ply")
            filebase["raw_label_filepath"] = label_filepath
            label_coords, label_colors, labels = load_ply_with_normals(label_filepath)
            if not np.allclose(coords, label_coords):
                raise ValueError("files doesn't have same coordinates")

            # adding instance label
            labels = labels[:, np.newaxis]
            empty_instance_label = np.full(labels.shape, -1)
            labels = np.hstack((labels, empty_instance_label))
            for instance in instance_db["segGroups"]:
                segments_occupied = np.array(instance["segments"])
                occupied_indices = np.isin(segments, segments_occupied)
                labels[occupied_indices, 1] = instance["id"]
            #############################################!!!
            points = np.hstack((points, labels))###[xyz, rgb, nxnynz, segment, semantic, instance]

            gt_data = points[:, -2] * 1000 + points[:, -1] + 1
            if self.vis:
                vis_semantic_path = os.path.join(self.save_dir / mode / 'vis_semantic')
                vis_instance_path = os.path.join(self.save_dir / mode / 'vis_instance')
                os.makedirs(vis_semantic_path, exist_ok=True)
                os.makedirs(vis_instance_path, exist_ok=True)
                sem_colors, inst_colors = np.zeros((points.shape[0], 3)), np.zeros((points.shape[0], 3))
                inst_color_map = get_evenly_distributed_colors(int(points[:, -1].max()+1))
                for point_idx in range(points.shape[0]):
                    if int(points[:, -2][point_idx]) in VALID_CLASS_IDS_20 and int(points[:, -2][point_idx])!=-1:
                        sem_colors[point_idx] = SCANNET_COLOR_MAP_20[int(points[:, -2][point_idx])]
                    if int(points[:, -1][point_idx])!=-1:
                        inst_colors[point_idx] = inst_color_map[int(points[:, -1][point_idx])]
                # write_ply(os.path.join(vis_semantic_path, f"{scene:04}_{sub_scene:02}.ply"), [points[:, 0:3], sem_colors.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
                # write_ply(os.path.join(vis_instance_path, f"{scene:04}_{sub_scene:02}.ply"), [points[:, 0:3], inst_colors.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
                write_point_cloud_in_ply(os.path.join(vis_semantic_path, f"{scene:04}_{sub_scene:02}.ply"), points[:, 0:3], sem_colors.astype(np.uint8))
                write_point_cloud_in_ply(os.path.join(vis_instance_path, f"{scene:04}_{sub_scene:02}.ply"), points[:, 0:3], inst_colors.astype(np.uint8))
        else:
            segments_test = filepath.parent#"../../data/raw/scannet_test_segments"
            # segments_test = "../../data/raw/scannet_test_segments"
            segment_indexes_filepath = filepath.name.replace(".ply", ".0.010000.segs.json")
            segments = self._read_json(f"{segments_test}/{segment_indexes_filepath}")
            segments = np.array(segments["segIndices"])
            # add segment id as additional feature
            segment_ids = np.unique(segments, return_inverse=True)[1]
            points = np.hstack((points, segment_ids[..., None]))

        if self.vis:
            vis_segment_path = os.path.join(self.save_dir / mode / 'vis_segment')
            os.makedirs(vis_segment_path, exist_ok=True)
            segment_colors = np.zeros((points.shape[0], 3))
            segment_color_map = colormap#get_evenly_distributed_colors(int(points[:, 9].max()+1))
            for point_idx in range(points.shape[0]):
                if int(points[:, 9][point_idx])!=-1:
                    segment_colors[point_idx] = segment_color_map[int(points[:, 9][point_idx])]
            # write_ply(os.path.join(vis_segment_path, f"{scene:04}_{sub_scene:02}.ply"), [points[:, 0:3], segment_colors.astype(np.uint8)], ['x', 'y', 'z', 'red', 'green', 'blue'])
            write_point_cloud_in_ply(os.path.join(vis_segment_path, f"{scene:04}_{sub_scene:02}.ply"), points[:, 0:3], segment_colors.astype(np.uint8))

        processed_filepath = (self.save_dir / mode / f"{scene:04}_{sub_scene:02}.npy")
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        if mode == "test":
            return filebase

        processed_gt_filepath = (self.save_dir/ "instance_gt" / mode / f"scene{scene:04}_{sub_scene:02}.txt")
        if not processed_gt_filepath.parent.exists():
            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
        filebase["instance_gt_filepath"] = str(processed_gt_filepath)

        filebase["color_mean"] = [
            float((features[:, 0] / 255).mean()),
            float((features[:, 1] / 255).mean()),
            float((features[:, 2] / 255).mean())]
        filebase["color_std"] = [
            float(((features[:, 0] / 255) ** 2).mean()),
            float(((features[:, 1] / 255) ** 2).mean()),
            float(((features[:, 2] / 255) ** 2).mean())]
        return filebase

    def compute_color_mean_std(self, train_database_path: str = "./data/processed/scannet/train_database.yaml"):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean**2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std]}
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)

    @logger.catch
    def fix_bugs_in_labels(self):
        logger.add(self.save_dir / "fixed_bugs_in_labels.log")
        found_wrong_labels = {
            tuple([270, 0]): 50,
            tuple([270, 2]): 50,
            tuple([384, 0]): 149}
        for scene, wrong_label in found_wrong_labels.items():
            scene, sub_scene = scene
            bug_file = (self.save_dir / "train" / f"{scene:04}_{sub_scene:02}.npy")
            points = np.load(bug_file)
            bug_mask = points[:, -1] != wrong_label
            points = points[bug_mask]
            np.save(bug_file, points)
            logger.info(f"Fixed {bug_file}")

    def _parse_scene_subscene(self, name):
        scene_match = re.match(r"scene(\d{4})_(\d{2})", name)
        return int(scene_match.group(1)), int(scene_match.group(2))


if __name__ == "__main__":
    # Fire(ScannetPreprocessing)
    ScannetPreprocessing()
