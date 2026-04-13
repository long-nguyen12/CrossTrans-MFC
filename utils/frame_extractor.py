import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import DATASET_CONFIG, VIDEO_DESCRIPTOR_CONFIG
import argparse
from PIL import Image

import cv2
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA
from transformers import CLIPImageProcessor, CLIPVisionModel
import time
import csv
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def reorder_and_rename_images(directory_path):
    directory = Path(directory_path)
    images = sorted(directory.glob("*.jpeg"), key=lambda p: p.stat().st_mtime)
    if not images:
        logging.info("No JPEG images found to rename in %s", directory_path)
        return

    # Use temporary names to avoid collisions when target names already exist.
    tmp_paths = []
    for idx, image_path in enumerate(images, start=1):
        tmp_path = image_path.with_name(f"__tmp_{idx:06d}.jpeg")
        image_path.rename(tmp_path)
        tmp_paths.append(tmp_path)

    for idx, tmp_path in enumerate(tmp_paths, start=1):
        tmp_path.rename(directory / f"{idx}.jpeg")

    # logging.info("Images have been renamed successfully.")


def _load_clip_vision(model_name):
    processor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPVisionModel.from_pretrained(model_name)
    model.eval()
    return processor, model


def _read_frames_at_indices(cap, indices):
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return frames


def _select_representative_frame(frames, processor, model, device):
    if not frames:
        return None
    inputs = processor(images=frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    return embeddings


def _select_representative_frame_spectral(
    frames,
    embeddings,
    n_clusters,
    plot_clusters=False,
    plot_path=None,
    plot_title=None,
):
    if not frames:
        return None
    if embeddings.shape[0] == 1:
        return frames[0]

    n_clusters = max(2, min(int(n_clusters), int(embeddings.shape[0])))
    n_samples = int(embeddings.shape[0])
    n_neighbors = max(1, min(10, n_samples - 1))
    labels = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        assign_labels="kmeans",
        random_state=0,
    ).fit_predict(embeddings.cpu().numpy())

    # labels = KMeans(
    #     n_clusters=n_clusters,
    #     random_state=0,
    #     n_init="auto",
    # ).fit_predict(embeddings.cpu().numpy())

    # Choose the medoid of the largest cluster.
    largest_label = max(set(labels), key=lambda l: (labels == l).sum())
    idxs = [i for i, l in enumerate(labels) if l == largest_label]
    cluster_emb = embeddings[idxs]
    centroid = cluster_emb.mean(dim=0, keepdim=True)
    distances = torch.cdist(cluster_emb, centroid).squeeze(1)
    best_local = int(torch.argmin(distances).item())

    if plot_clusters:
        saved_dir = "../plots/"
        try:
            reduced = PCA(n_components=2).fit_transform(embeddings.cpu().numpy())
            fig, ax = plt.subplots(figsize=(6, 5))
            scatter = ax.scatter(
                reduced[:, 0],
                reduced[:, 1],
                c=labels,
                cmap="tab10",
                alpha=0.7,
                s=40,
                edgecolor="k",
                linewidth=0.2,
            )
            medoid_coords = reduced[idxs[best_local]]
            ax.scatter(
                medoid_coords[0],
                medoid_coords[1],
                c="red",
                s=140,
                marker="*",
                edgecolor="k",
                linewidth=0.8,
                label="selected medoid",
            )
            ax.legend(fontsize=10)
            ax.set_title(plot_title or "Cluster distribution", fontsize=12)
            ax.set_xlabel("PC 1", fontsize=10)
            ax.set_ylabel("PC 2", fontsize=10)
            ax.grid(True, alpha=0.3)

            if plot_path is None:
                plot_path = f"{saved_dir}/spectral_clustering/cluster_distribution_{time.time()}.png"
            os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            logging.warning("Failed to plot cluster distribution: %s", e)

    return frames[idxs[best_local]]


def clip_chunk_keyframes_extraction(
    model,
    processor,
    video_file_path,
    chunk_count=10,
    samples_per_chunk=8,
    spectral_clusters=2,
    output_dir=None,
    device=None,
):
    video_path = Path(video_file_path)
    if output_dir is None:
        target_path = video_path.parent / video_path.stem
    else:
        target_path = Path(output_dir) / video_path.stem

    # if target_path.exists() and len(list(target_path.glob("*.jpeg"))) >= chunk_count:
    #     logging.info("Keyframes already extracted and present in %s", target_path)
    #     return str(target_path)

    cap = cv2.VideoCapture(str(video_file_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_file_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has no frames: {video_file_path}")

    chunk_count = min(chunk_count, total_frames)
    chunk_size = total_frames / chunk_count

    print(f"Using device: {device}")

    target_path.mkdir(parents=True, exist_ok=True)
    saved = 0

    for chunk_idx in range(chunk_count):
        start = int(chunk_idx * chunk_size)
        end = int(min(total_frames, (chunk_idx + 1) * chunk_size))
        if end <= start:
            continue

        if samples_per_chunk <= 1 or (end - start) <= 1:
            indices = [start]
        else:
            step = max(1, (end - start) // samples_per_chunk)
            indices = list(range(start, end, step))[:samples_per_chunk]

        frames = _read_frames_at_indices(cap, indices)
        embeddings = _select_representative_frame(frames, processor, model, device)
        if embeddings is None:
            continue

        best_frame = _select_representative_frame_spectral(
            frames, embeddings, spectral_clusters
        )

        if best_frame is None:
            continue

        saved += 1
        out_path = target_path / f"{saved}.jpeg"
        out_bgr = cv2.cvtColor(best_frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), out_bgr)

    cap.release()

    reorder_and_rename_images(str(target_path))
    # logging.info(
    #     "video %s: CLIP chunk keyframes extracted successfully", video_path.stem
    # )
    return str(target_path)


def process_folder_videos(args):
    dataset_root = Path(f"{DATASET_CONFIG['root_dir']}").resolve()

    output_root = Path(f"{DATASET_CONFIG['root_dir']}").resolve()

    test_annotation = dataset_root / DATASET_CONFIG["annotation"][args.split]
    test_video_dir = dataset_root / DATASET_CONFIG["video_dir"][args.split]

    test_output_dir = output_root / DATASET_CONFIG["output_dir"][args.split]

    test_output_dir.mkdir(parents=True, exist_ok=True)

    with open(test_annotation, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model_name = VIDEO_DESCRIPTOR_CONFIG.get(
        "clip_model_name", "openai/clip-vit-base-patch32"
    )
    processor, model = _load_clip_vision(model_name)
    model.to(device)
    with open("clipchunk_time.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "time", "video_length"])
        for i, video_id in tqdm(
            enumerate(video_ids), total=len(video_ids), desc="Processing videos"
        ):
            try:
                video_file = None
                for ext in [".mp4", ".mkv"]:
                    potential_file = os.path.join(test_video_dir, f"{video_id}{ext}")
                    if os.path.exists(potential_file):
                        video_file = potential_file
                        break

                if not video_file:
                    logging.warning(f"No video file found for ID: {video_id}")
                    continue

                data_folder = os.path.join(test_video_dir, str(video_id))

                if not os.path.exists(data_folder):
                    os.makedirs(data_folder)

                try:
                    chunk_count = VIDEO_DESCRIPTOR_CONFIG.get("chunk_count", 10)
                    samples_per_chunk = VIDEO_DESCRIPTOR_CONFIG.get(
                        "clip_samples_per_chunk", 8
                    )
                    spectral_clusters = VIDEO_DESCRIPTOR_CONFIG.get(
                        "spectral_clusters_per_chunk", 2
                    )
                    start_time = time.time()
                    _ = clip_chunk_keyframes_extraction(
                        model,
                        processor,
                        video_file,
                        chunk_count=chunk_count,
                        samples_per_chunk=samples_per_chunk,
                        spectral_clusters=spectral_clusters,
                        output_dir=test_output_dir,
                        device=device,
                    )
                    end_time = time.time() - start_time
                    video_length = _get_video_length(video_file)
                    writer.writerow([video_id, end_time, video_length])
                except Exception as e:
                    logging.error(
                        f"Failed to extract keyframes for video {video_id}: {str(e)}"
                    )
                    continue
            except Exception as e:
                logging.error(f"Error processing video {video_id}: {str(e)}")
                continue


from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter


def _get_video_length(video_file):
    cap = cv2.VideoCapture(video_file)
    video_length = 0
    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
    return video_length


def katna_keyframes_extraction(
    video_file_path, no_of_frames_to_returned, output_dir=None
):
    vd = Video()

    video_dir, video_name = os.path.split(video_file_path)
    video_base_name = os.path.splitext(video_name)[0]
    if output_dir is None:
        target_path = os.path.join(video_dir, video_base_name)
    else:
        if not os.path.isdir(output_dir):
            raise FileNotFoundError(
                f"Output directory does not exist (no auto-create): {output_dir}"
            )
        target_path = os.path.join(output_dir, video_base_name)

    if (
        os.path.exists(target_path)
        and len([f for f in os.listdir(target_path) if f.endswith(".jpeg")])
        >= no_of_frames_to_returned
    ):
        logging.info(f"Keyframes already extracted and present in {target_path}")
        return target_path, 0

    disk_writer = KeyFrameDiskWriter(location=target_path)

    logging.info(f"Input video file path = {video_file_path}")
    start = time.time()
    vd.extract_video_keyframes(
        no_of_frames=no_of_frames_to_returned,
        file_path=video_file_path,
        writer=disk_writer,
    )
    end = time.time() - start
    logging.info(f"video {video_base_name}：Keyframes extracted successfully")

    reorder_and_rename_images(target_path)

    return target_path, end


def katna_process_folder(args):
    dataset_root = Path(f"{DATASET_CONFIG['root_dir']}").resolve()
    # Writable output root
    output_root = Path(f"{DATASET_CONFIG['root_dir']}").resolve()

    test_annotation = dataset_root / DATASET_CONFIG["annotation"][args.split]
    test_video_dir = dataset_root / DATASET_CONFIG["video_dir"][args.split]

    test_output_dir = output_root / "katna" / DATASET_CONFIG["output_dir"][args.split]

    test_output_dir.mkdir(parents=True, exist_ok=True)

    with open(test_annotation, "r") as f:
        video_ids = [line.strip() for line in f if line.strip()]

    with open("katna_time.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "time", "video_length"])
        for i, video_id in tqdm(
            enumerate(video_ids), total=len(video_ids), desc="Processing videos"
        ):
            try:
                logging.info(f"Processing video {i + 1}/{len(video_ids)}: {video_id}")
                video_file = None
                for ext in [".mp4", ".mkv"]:
                    potential_file = os.path.join(test_video_dir, f"{video_id}{ext}")
                    if os.path.exists(potential_file):
                        video_file = potential_file
                        break

                if not video_file:
                    logging.warning(f"No video file found for ID: {video_id}")
                    continue

                data_folder = os.path.join(test_video_dir, str(video_id))

                if not os.path.exists(data_folder):
                    os.makedirs(data_folder)

                cap = cv2.VideoCapture(video_file)
                resolution_too_large = False
                if cap.isOpened():
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    cap.release()
                    if width * height > 3840 * 2160:
                        logging.warning(f"Skipping video {video_id} due to large resolution: {width}x{height}")
                        resolution_too_large = True

                if resolution_too_large:
                    continue

                logging.info(f"Extracting keyframes for video: {video_id}")
                try:
                    chunk_count = VIDEO_DESCRIPTOR_CONFIG.get("chunk_count", 10)

                    target_path, time_taken = katna_keyframes_extraction(
                        video_file,
                        no_of_frames_to_returned=chunk_count,
                        output_dir=test_output_dir,
                    )

                    video_length = _get_video_length(video_file)
                    writer.writerow([video_id, time_taken, video_length])
                    logging.info(
                        f"Keyframes extracted successfully for video: {video_id}"
                    )
                except Exception as e:
                    logging.error(
                        f"Failed to extract keyframes for video {video_id}: {str(e)}"
                    )
                    continue
            except Exception as e:
                logging.error(f"Error processing video {video_id}: {str(e)}")
                continue


def _extract_image_features(
    model,
    processor,
    image,
    device,
):
    try:
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            if getattr(model, "feature_method", None):
                method = getattr(model, model.feature_method)
                features = method(**inputs)
            elif hasattr(model, "get_image_features"):
                features = model.get_image_features(**inputs)
                features = features.pooler_output
            else:
                outputs = model(**inputs)
                features = outputs.last_hidden_state[:, 0, :]

        model_type = getattr(
            model,
            "vision_type",
            getattr(getattr(model, "config", None), "model_type", None),
        )
        if model_type in ["clip", "siglip"]:
            features = features / features.norm(dim=-1, keepdim=True)

        return features.squeeze(0)  # [T, D] or [D]
    except Exception as e:
        print(e)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--data-root",
        type=str,
        default="../data/TRUE_Dataset",
        help="Dataset root directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train_val",
        choices=["train_val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/video_cache",
        help="Dir with pre-extracted video feature .pt files",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="katna",
        choices=["katna", "clip"],
        help="Method to extract keyframes",
    )
    parser.add_argument(
        "--limit",
        type=any,
        default=None,
        help="Number of videos to extract",
    )
    args = parser.parse_args()
    if args.method == "katna":
        katna_process_folder(args)
    elif args.method == "clip":
        process_folder_videos(args)
