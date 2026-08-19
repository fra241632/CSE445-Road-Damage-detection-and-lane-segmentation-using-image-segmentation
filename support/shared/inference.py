"""
support/shared/inference.py
---------------------------
Unified inference engine for Road Surface Crack Detection.
Supports both single Image and Video inputs using trained U-Net (or DeepLabv3) models.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any, List

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from support.shared.unet import UNet
from support.shared.transforms import get_val_transforms


class CrackPredictor:
    """
    Inference wrapper for Road Surface Crack Segmentation.
    Works seamlessly on both images and video files.
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        model_type: str = "unet",
        device: Optional[str] = None,
        img_size: Tuple[int, int] = (config.IMG_HEIGHT, config.IMG_WIDTH),
    ):
        """
        Args:
            checkpoint_path: Path to .pth weights. If None, resolves default crack_unet_run1 checkpoint.
            model_type: 'unet'
            device: 'cuda', 'cpu', or None (auto-detect)
            img_size: (height, width) for model input tensor resizing
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.img_size = img_size
        self.model_type = model_type.lower()
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self.model = self._load_model()
        self.transform = get_val_transforms(self.img_size)

    def _resolve_checkpoint(self, path: Optional[Union[str, Path]]) -> Path:
        if path is not None:
            p = Path(path).resolve()
            if p.exists():
                return p
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        if self.model_type == "deeplabv3":
            candidates = [
                REPO_ROOT / "experiments" / "crack_stage2_deeplabv3_finetuned" / "best_model.pth",
                REPO_ROOT / "experiments" / "crack_stage1_deeplabv3" / "best_model.pth",
                REPO_ROOT / "experiments" / "crack_deeplabv3_run1" / "best_model.pth",
                Path("experiments/crack_stage2_deeplabv3_finetuned/best_model.pth").resolve(),
                Path("experiments/crack_stage1_deeplabv3/best_model.pth").resolve(),
                Path("experiments/crack_deeplabv3_run1/best_model.pth").resolve(),
            ]
        else:
            candidates = [
                REPO_ROOT / "experiments" / "crack_stage2_unet_finetuned" / "best_model.pth",
                REPO_ROOT / "experiments" / "crack_stage1_unet" / "best_model.pth",
                REPO_ROOT / "experiments" / "crack_unet_run1" / "best_model.pth",
                Path("experiments/crack_stage2_unet_finetuned/best_model.pth").resolve(),
                Path("experiments/crack_stage1_unet/best_model.pth").resolve(),
                Path("experiments/crack_unet_run1/best_model.pth").resolve(),
                Path("../experiments/crack_stage2_unet_finetuned/best_model.pth").resolve(),
                Path("../experiments/crack_unet_run1/best_model.pth").resolve(),
                Path("/content/Road_Damage_Project/experiments/crack_stage2_unet_finetuned/best_model.pth"),
                Path("/content/Road_Damage_Project/experiments/crack_unet_run1/best_model.pth"),
            ]
        for c in candidates:
            if c.exists():
                return c.resolve()

        raise FileNotFoundError(
            f"Default model checkpoint for {self.model_type} not found. Please specify checkpoint_path or train a model first."
        )


    def _load_model(self) -> torch.nn.Module:
        if self.model_type == "deeplabv3":
            from support.shared.deeplabv3 import DeepLabV3Segmentation
            model = DeepLabV3Segmentation(
                in_channels=config.CRACK_DEEPLABV3.get("in_channels", 3),
                out_channels=config.CRACK_DEEPLABV3.get("out_channels", 1),
                backbone=config.CRACK_DEEPLABV3.get("backbone", "resnet50"),
                pretrained=False,
            )
        else:
            model = UNet(
                in_channels=config.CRACK_UNET.get("in_channels", 3),
                out_channels=config.CRACK_UNET.get("out_channels", 1),
                base_features=config.CRACK_UNET.get("base_features", 32),
            )

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()
        return model

    def predict_frame(
        self,
        frame_rgb: np.ndarray,
        threshold: float = 0.20,
        crack_color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 1.0,
        overlay_mode: str = "both",  # 'both' (mask + bboxes), 'bbox' (bboxes only), 'mask' (mask only)
        draw_boxes: bool = True,     # Enable bounding boxes around detected cracks
        min_box_area: int = 20,      # Minimum contour area in pixels to draw a bounding box
        box_thickness: int = 2,      # -1 / cv2.FILLED for full filled solid box; > 0 for outline
        box_color: Optional[Tuple[int, int, int]] = (255, 255, 0),
        draw_box_labels: bool = False,
        morph_close: bool = True,    # Bridge fragmented big crack contours
    ) -> Dict[str, Any]:
        """
        Run inference on a single RGB frame (numpy array [H, W, 3]).

        Args:
            frame_rgb: Input image in RGB format.
            threshold: Probability threshold for crack detection.
            crack_color: RGB tuple for crack highlight (default solid white: (255, 255, 255)).
            alpha: Opacity factor for segmentation mask blending (1.0 = solid / no opacity).
            overlay_mode: Visualization mode:
                          - 'both': Draw both solid red full filled box AND solid white mask.
                          - 'bbox' or 'box': Draw solid red full filled box around cracks only.
                          - 'mask': Draw solid white segmentation mask only.
            draw_boxes: Whether to draw bounding boxes around detected crack contours.
            min_box_area: Minimum pixel area for a crack blob to get a bounding box (filters noise).
            box_thickness: Box line thickness (-1 / cv2.FILLED for full solid box).
            box_color: RGB tuple for the bounding box (default red: (255, 0, 0)).
            draw_box_labels: Whether to render confidence badges above bounding boxes.

        Returns dict:
            prob_map: float32 [H, W] probability in [0, 1]
            binary_mask: uint8 [H, W] (0 or 1)
            overlay: RGB image [H, W, 3] with crack segmentation overlay / bounding boxes
            crack_pixels: count of positive crack pixels
            total_pixels: total pixel count
            crack_pct: percentage of crack coverage
            max_prob: max predicted probability
            mean_prob: mean predicted probability
            boxes: list of dicts with bounding box coordinates and statistics
            num_boxes: total number of detected crack bounding boxes
        """
        orig_h, orig_w = frame_rgb.shape[:2]

        augmented = self.transform(image=frame_rgb)
        input_tensor = augmented["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(input_tensor)

        prob_256 = pred.squeeze().cpu().numpy()

        # Resize probability back to original frame dimensions
        prob_orig = cv2.resize(
            prob_256, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
        binary_mask = (prob_orig > threshold).astype(np.uint8)

        # Morphological closing to connect fragmented/wide crack components
        if morph_close:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            contour_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        else:
            contour_mask = binary_mask

        # Extract bounding boxes from connected crack components
        contours, _ = cv2.findContours(
            contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_box_area:
                x, y, w, h = cv2.boundingRect(cnt)
                box_prob = float(prob_orig[y : y + h, x : x + w].max()) if w > 0 and h > 0 else 0.0
                boxes.append({
                    "box": (int(x), int(y), int(w), int(h)),
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "area": float(area),
                    "max_prob": box_prob,
                })

        blended = frame_rgb.copy()

        # 1. Render Red Solid Full Bounding Boxes (if requested)
        show_boxes = draw_boxes and (overlay_mode in ("bbox", "box", "both", "mask_and_bbox", "mask_and_box"))
        if show_boxes:
            b_color = box_color if box_color is not None else (255, 0, 0)
            for b in boxes:
                x, y, w, h = b["x"], b["y"], b["w"], b["h"]
                cv2.rectangle(blended, (x, y), (x + w, y + h), b_color, box_thickness)
                if draw_box_labels:
                    label = f"Crack {b['max_prob']:.0%}"
                    (lbl_w, lbl_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    lbl_y = max(lbl_h + 4, y - 2)
                    cv2.rectangle(
                        blended,
                        (x, lbl_y - lbl_h - 4),
                        (x + lbl_w + 4, lbl_y + 2),
                        (20, 20, 20),
                        -1,
                    )
                    cv2.putText(
                        blended,
                        label,
                        (x + 2, lbl_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        b_color,
                        1,
                        cv2.LINE_AA,
                    )

        # 2. Render White Segmentation Mask ON TOP of the red box (if requested)
        show_mask = overlay_mode in ("mask", "both", "mask_and_bbox", "mask_and_box")
        if show_mask:
            overlay_mask = np.zeros_like(frame_rgb)
            overlay_mask[binary_mask == 1] = crack_color
            mask_idx = binary_mask == 1
            if np.any(mask_idx):
                if alpha >= 1.0:
                    blended[mask_idx] = overlay_mask[mask_idx]
                else:
                    blended[mask_idx] = cv2.addWeighted(
                        blended, 1.0 - alpha, overlay_mask, alpha, 0
                    )[mask_idx]

        crack_pixels = int(binary_mask.sum())
        total_pixels = int(binary_mask.size)
        crack_pct = (crack_pixels / total_pixels) * 100.0

        return {
            "prob_map": prob_orig,
            "binary_mask": binary_mask,
            "overlay": blended,
            "crack_pixels": crack_pixels,
            "total_pixels": total_pixels,
            "crack_pct": crack_pct,
            "max_prob": float(prob_orig.max()),
            "mean_prob": float(prob_orig.mean()),
            "boxes": boxes,
            "num_boxes": len(boxes),
        }

    def predict_image(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.20,
        crack_color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 1.0,
        overlay_mode: str = "both",
        draw_boxes: bool = True,
        min_box_area: int = 20,
        box_thickness: int = 2,
        box_color: Optional[Tuple[int, int, int]] = (255, 255, 0),
        draw_box_labels: bool = False,
        morph_close: bool = True,
        save_plot: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run inference on a single image file.
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise FileNotFoundError(f"Could not open image file: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        result = self.predict_frame(
            img_rgb,
            threshold=threshold,
            crack_color=crack_color,
            alpha=alpha,
            overlay_mode=overlay_mode,
            draw_boxes=draw_boxes,
            min_box_area=min_box_area,
            box_thickness=box_thickness,
            box_color=box_color,
            draw_box_labels=draw_box_labels,
            morph_close=morph_close,
        )
        result["image_path"] = str(image_path)
        result["original_image"] = img_rgb

        if output_path is None:
            p = Path(image_path)
            output_path = p.parent / f"{p.stem}_crack_masked.jpg"

        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if save_plot:
            self.save_comparison_plot(
                img_rgb,
                result["binary_mask"],
                result["overlay"],
                out_p,
                result["crack_pct"],
                num_boxes=result.get("num_boxes", 0),
            )
        else:
            out_bgr = cv2.cvtColor(result["overlay"], cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_p), out_bgr)
        result["saved_to"] = str(out_p)
        result["output_path"] = str(out_p)

        return result

    def predict_video(
        self,
        video_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.20,
        crack_color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 1.0,
        overlay_mode: str = "both",
        draw_boxes: bool = True,
        min_box_area: int = 20,
        box_thickness: int = 2,
        box_color: Optional[Tuple[int, int, int]] = (255, 255, 0),
        draw_box_labels: bool = False,
        morph_close: bool = True,
        show_hud: bool = False,
        hud: Optional[bool] = None,
        layout: str = "overlay",  # 'overlay', 'side_by_side', or 'mask_only'
        show_live_window: bool = False,
        max_frames: Optional[int] = None,
        show_progress: bool = True,
        sample_interval: int = 30,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run inference on a video file frame-by-frame and generate an annotated video.

        Args:
            video_path: Path to input video file.
            output_path: Path to save processed output video.
            threshold: Probability threshold for crack detection.
            crack_color: RGB tuple for crack highlight (default solid white: (255, 255, 255)).
            alpha: Transparency factor for crack overlay (1.0 = solid / no opacity).
            box_color: RGB tuple for the bounding box (default red: (255, 0, 0)).
            hud: Whether to render real-time statistics HUD on video frames.
            show_hud: Alias for hud.
            layout: Video composition style:
                    - 'overlay': Crack mask blended over original video.
                    - 'side_by_side': Original video (left) + Masked video (right).
                    - 'mask_only': Pure binary crack mask video.
            show_live_window: If True, opens an interactive desktop window showing the masked video playing in real-time.
            max_frames: Stop after N frames (useful for quick previews/tests).
            show_progress: Print processing progress in console.
            sample_interval: Collect a sample frame every N frames for notebook visualization.

        Returns dict of video summary statistics, sample frames, and video path.
        """
        if hud is None:
            hud = show_hud

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video file: {video_path}")

        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if max_frames is not None and max_frames > 0:
            target_frames = min(total_video_frames, max_frames)
        else:
            target_frames = total_video_frames

        # Determine output frame size based on layout
        if layout == "side_by_side":
            out_w, out_h = orig_w * 2, orig_h
        else:
            out_w, out_h = orig_w, orig_h

        # Output video writer setup
        if output_path is None:
            p = Path(video_path)
            output_path = p.parent / f"{p.stem}_crack_masked.mp4"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, input_fps, (out_w, out_h))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(str(output_path), fourcc, input_fps, (out_w, out_h))

        frame_idx = 0
        frames_with_cracks = 0
        crack_percentages: List[float] = []
        sample_frames: List[Dict[str, Any]] = []

        start_time = time.time()

        if show_live_window:
            cv2.namedWindow("Road Crack Detection - Video Playback", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Road Crack Detection - Video Playback", min(1280, out_w), min(720, out_h))

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            res = self.predict_frame(
                frame_rgb,
                threshold=threshold,
                crack_color=crack_color,
                alpha=alpha,
                overlay_mode=overlay_mode,
                draw_boxes=draw_boxes,
                min_box_area=min_box_area,
                box_thickness=box_thickness,
                box_color=box_color,
                draw_box_labels=draw_box_labels,
            )
            t_infer = (time.perf_counter() - t0) * 1000.0  # ms
            instant_fps = 1000.0 / max(t_infer, 1e-3)

            crack_pct = res["crack_pct"]
            crack_percentages.append(crack_pct)
            has_crack = crack_pct > 0.05
            if has_crack:
                frames_with_cracks += 1

            # Prepare annotated frame according to layout
            if layout == "side_by_side":
                left_panel = frame_rgb.copy()
                right_panel = res["overlay"].copy()
                if hud:
                    right_panel = self._draw_hud(
                        right_panel,
                        frame_idx=frame_idx,
                        total_frames=target_frames,
                        fps=instant_fps,
                        crack_pct=crack_pct,
                        has_crack=has_crack,
                    )
                composed_rgb = np.hstack([left_panel, right_panel])
            elif layout == "mask_only":
                mask_3c = cv2.cvtColor((res["binary_mask"] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
                composed_rgb = mask_3c
            else:  # 'overlay'
                composed_rgb = res["overlay"].copy()
                if hud:
                    composed_rgb = self._draw_hud(
                        composed_rgb,
                        frame_idx=frame_idx,
                        total_frames=target_frames,
                        fps=instant_fps,
                        crack_pct=crack_pct,
                        has_crack=has_crack,
                    )

            composed_bgr = cv2.cvtColor(composed_rgb, cv2.COLOR_RGB2BGR)
            writer.write(composed_bgr)

            # Live desktop window preview
            if show_live_window:
                cv2.imshow("Road Crack Detection - Video Playback", composed_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key in [27, ord('q')]:  # ESC or q
                    print("\n[!] Live preview stopped by user.")
                    break

            if frame_idx % sample_interval == 1 or frame_idx == target_frames:
                sample_frames.append({
                    "frame_idx": frame_idx,
                    "original": frame_rgb,
                    "annotated": composed_rgb,
                    "mask": res["binary_mask"],
                    "crack_pct": crack_pct,
                    "boxes": res.get("boxes", []),
                    "num_boxes": res.get("num_boxes", 0),
                })

            if show_progress and (frame_idx % 20 == 0 or frame_idx == target_frames):
                elapsed = time.time() - start_time
                avg_fps = frame_idx / max(elapsed, 1e-5)
                progress_pct = (frame_idx / max(target_frames, 1)) * 100.0
                print(
                    f"\rProcessing Video: Frame {frame_idx}/{target_frames} "
                    f"({progress_pct:.1f}%) | Avg FPS: {avg_fps:.1f} | Crack %: {crack_pct:.2f}%",
                    end="",
                    flush=True,
                )

        cap.release()
        if writer is not None:
            writer.release()
        if show_live_window:
            cv2.destroyAllWindows()

        total_time = time.time() - start_time
        avg_processing_fps = frame_idx / max(total_time, 1e-5)
        if show_progress:
            print(f"\n[OK] Video processing complete in {total_time:.2f}s ({avg_processing_fps:.1f} FPS)!")
            print(f"[OK] Masked video saved to: {output_path}")

        avg_crack_pct = float(np.mean(crack_percentages)) if crack_percentages else 0.0
        max_crack_pct = float(np.max(crack_percentages)) if crack_percentages else 0.0

        return {
            "input_path": str(video_path),
            "output_path": str(output_path),
            "total_frames_processed": frame_idx,
            "frames_with_cracks": frames_with_cracks,
            "crack_frame_ratio": (frames_with_cracks / max(frame_idx, 1)) * 100.0,
            "avg_crack_pct": avg_crack_pct,
            "max_crack_pct": max_crack_pct,
            "total_time_sec": total_time,
            "avg_fps": avg_processing_fps,
            "crack_percentages_series": crack_percentages,
            "sample_frames": sample_frames,
            "layout": layout,
        }

    def process(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Auto-detects whether input is an Image or Video and executes corresponding pipeline.
        """
        p = Path(input_path)
        ext = p.suffix.lower()
        if ext in self.VIDEO_EXTENSIONS:
            return self.predict_video(p, output_path=output_path, **kwargs)
        elif ext in self.IMAGE_EXTENSIONS:
            return self.predict_image(p, output_path=output_path, **kwargs)
        else:
            # Try image first, then video
            try:
                return self.predict_image(p, output_path=output_path, **kwargs)
            except Exception:
                return self.predict_video(p, output_path=output_path, **kwargs)

    @staticmethod
    def _draw_hud(
        img_rgb: np.ndarray,
        frame_idx: int,
        total_frames: int,
        fps: float,
        crack_pct: float,
        has_crack: bool,
    ) -> np.ndarray:
        """
        Renders a subtle corner status tag on the frame without obscuring video content.
        """
        h, w = img_rgb.shape[:2]
        canvas = img_rgb.copy()

        # Compact corner badge in top-left
        badge_w, badge_h = int(min(360, w * 0.4)), int(max(32, h * 0.05))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (10 + badge_w, 10 + badge_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

        badge_color = (255, 60, 60) if has_crack else (60, 220, 100)
        status_text = f"Crack: {crack_pct:.1f}% | FPS: {fps:.0f}"
        
        font_scale = max(0.4, h / 1080.0 * 0.7)
        thickness = max(1, int(font_scale * 2))

        cv2.putText(
            canvas,
            status_text,
            (20, 10 + int(badge_h * 0.7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            badge_color,
            thickness,
            cv2.LINE_AA,
        )

        return canvas

    @staticmethod
    def play_video_in_notebook(video_path: Union[str, Path], width: int = 700):
        """
        Embeds an HTML5 playable video player directly inside Jupyter Notebook or Google Colab.
        """
        import base64
        from IPython.display import HTML, display

        p = Path(video_path)
        if not p.exists():
            print(f"[Error] Video file not found: {video_path}")
            return

        with open(p, "rb") as f:
            video_bytes = f.read()

        b64 = base64.b64encode(video_bytes).decode("ascii")
        html_code = f"""
        <div style="text-align: center; margin: 15px 0;">
            <video width="{width}" controls autoplay loop style="border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                <source src="data:video/mp4;base64,{b64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
            <p style="color: #666; font-size: 12px; margin-top: 5px;">Playing: {p.name}</p>
        </div>
        """
        display(HTML(html_code))

    @staticmethod
    def save_comparison_plot(
        original_rgb: np.ndarray,
        mask: np.ndarray,
        overlay_rgb: np.ndarray,
        save_path: Path,
        crack_pct: float,
        num_boxes: Optional[int] = None,
    ):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(original_rgb)
        plt.title("Original Road Image")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="gray")
        plt.title("Predicted Crack Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(overlay_rgb)
        title_box = f", {num_boxes} boxes" if num_boxes is not None else ""
        plt.title(f"Crack Detection ({crack_pct:.2f}%{title_box})")
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(str(save_path), bbox_inches="tight", dpi=150)
        plt.close()


def play_video_in_notebook(video_path: Union[str, Path], width: int = 700):
    """
    Top-level helper to embed an HTML5 playable video in Jupyter / Colab.
    """
    CrackPredictor.play_video_in_notebook(video_path, width=width)

