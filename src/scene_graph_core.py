"""
SPATIALINTEL – Step 1 Core Module
---------------------------------

Goal of this file:
    1. Run a pretrained YOLO model on an input image.
    2. Convert detections into structured SceneObject nodes.
    3. Build a simple 2D SceneGraph.
    4. Compute basic spatial relations: left_of, right_of, above, below, near.
    5. Print and optionally export the scene graph.

This will be extended later to:
    - Use video frames instead of a single image.
    - Use 3D positions from NeRF / Gaussian Splatting.
    - Feed the scene graph into an LLM for language reasoning.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import List, Tuple, Dict
import math
import json
import os

import cv2
from ultralytics import YOLO
import numpy as np


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class SceneObject:
    """
    Represents one detected object in the scene.

    bbox is in pixel coordinates:
        (x_min, y_min, x_max, y_max)
    """
    obj_id: str
    label: str
    bbox: Tuple[int, int, int, int]
    confidence: float

    def center(self) -> Tuple[float, float]:
        """Return (cx, cy) center of the bounding box."""
        x_min, y_min, x_max, y_max = self.bbox
        cx = (x_min + x_max) / 2.0
        cy = (y_min + y_max) / 2.0
        return cx, cy

    def width(self) -> int:
        x_min, _, x_max, _ = self.bbox
        return x_max - x_min

    def height(self) -> int:
        _, y_min, _, y_max = self.bbox
        return y_max - y_min


@dataclass
class SceneGraph:
    """
    Simple 2D scene graph.

    objects: list of SceneObject nodes
    relations: list of dicts with:
        { "subj": obj_id, "rel": relation_type, "obj": obj_id }
    """
    objects: List[SceneObject] = field(default_factory=list)
    relations: List[Dict] = field(default_factory=list)

    def add_object(self, obj: SceneObject) -> None:
        self.objects.append(obj)

    def compute_spatial_relations(
        self,
        near_threshold: float = 150.0,
        vertical_tolerance: float = 30.0
    ) -> None:
        """
        Compute pairwise spatial relations between all objects.

        Relations:
            - left_of / right_of   (based on x centers)
            - above / below        (based on y centers)
            - near                 (based on Euclidean distance)
        """
        self.relations.clear()

        for i in range(len(self.objects)):
            for j in range(len(self.objects)):
                if i == j:
                    continue

                A = self.objects[i]
                B = self.objects[j]

                cxA, cyA = A.center()
                cxB, cyB = B.center()

                # Horizontal relation
                if cxA < cxB:
                    horiz_rel = "left_of"
                else:
                    horiz_rel = "right_of"

                # Vertical relation
                if cyA < cyB - vertical_tolerance:
                    vert_rel = "above"
                elif cyA > cyB + vertical_tolerance:
                    vert_rel = "below"
                else:
                    vert_rel = None  # roughly same vertical level

                # Distance for "near" relation
                dist = math.dist((cxA, cyA), (cxB, cyB))
                is_near = dist < near_threshold

                # Add horizontal relation
                self.relations.append({
                    "subj": A.obj_id,
                    "rel": horiz_rel,
                    "obj": B.obj_id
                })

                # Add vertical relation if meaningful
                if vert_rel is not None:
                    self.relations.append({
                        "subj": A.obj_id,
                        "rel": vert_rel,
                        "obj": B.obj_id
                    })

                # Add near relation
                if is_near:
                    self.relations.append({
                        "subj": A.obj_id,
                        "rel": "near",
                        "obj": B.obj_id
                    })

    def to_dict(self) -> Dict:
        return {
            "objects": [asdict(o) for o in self.objects],
            "relations": self.relations,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def pretty_print(self) -> None:
        """Print objects and relations in a human-readable format."""
        print("\n===== SCENE OBJECTS =====")
        for obj in self.objects:
            print(
                f"- {obj.obj_id}: {obj.label} "
                f"| bbox={obj.bbox} | conf={obj.confidence:.2f}"
            )

        print("\n===== SPATIAL RELATIONS =====")
        for r in self.relations:
            print(f"{r['subj']}  --{r['rel']}-->  {r['obj']}")
        print("============================\n")


# -----------------------------
# YOLO-based detection
# -----------------------------

def run_yolo_detection(
    image_path: str,
    model_path: str = "yolov8n.pt",
    conf_threshold: float = 0.3
) -> List[SceneObject]:
    """
    Run a pretrained YOLO model on an image and return a list of SceneObject.

    Args:
        image_path: path to input image
        model_path: YOLO model weights (e.g. yolov8n.pt, yolov8s.pt, etc.)
        conf_threshold: minimum confidence to keep a detection

    Returns:
        List[SceneObject]
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[INFO] Loading YOLO model: {model_path}")
    model = YOLO(model_path)

    print(f"[INFO] Running inference on: {image_path}")
    results = model(image_path)[0]

    scene_objects: List[SceneObject] = []

    for idx, box in enumerate(results.boxes):
        cls_id = int(box.cls.item())
        label = model.names[cls_id]
        conf = float(box.conf.item())
        if conf < conf_threshold:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        obj = SceneObject(
            obj_id=f"obj_{idx+1}",
            label=label,
            bbox=(int(x1), int(y1), int(x2), int(y2)),
            confidence=conf
        )
        scene_objects.append(obj)

    print(f"[INFO] Total objects detected (after filtering): {len(scene_objects)}")
    return scene_objects


def visualize_detections(
    image_path: str,
    objects: List[SceneObject],
    output_path: str = "outputs/detections_overlay.jpg"
) -> None:
    """
    Draw bounding boxes and labels on the image and save the result.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    for obj in objects:
        x1, y1, x2, y2 = obj.bbox
        # draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # put label
        text = f"{obj.label} {obj.confidence:.2f}"
        cv2.putText(
            img, text, (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )

    cv2.imwrite(output_path, img)
    print(f"[INFO] Saved detection visualization to: {output_path}")


# -----------------------------
# Main entrypoint
# -----------------------------

def main():
    # TODO: later we can read this from CLI args or config file
    image_path = r"C:\Users\Sriman Rakshan N\Downloads\demodemo.jpg"  # Replace with your own image
    model_path = "yolov8n.pt"        # Ensure this model is downloaded by ultralytics
    graph_json_path = "outputs/scene_graph.json"
    vis_output_path = "outputs/detections_overlay.jpg"

    # 1) Run YOLO and get objects
    objects = run_yolo_detection(
        image_path=image_path,
        model_path=model_path,
        conf_threshold=0.3
    )

    # 2) Build scene graph
    scene_graph = SceneGraph()
    for obj in objects:
        scene_graph.add_object(obj)

    # 3) Compute spatial relations (2D)
    scene_graph.compute_spatial_relations(
        near_threshold=180.0,
        vertical_tolerance=30.0
    )

    # 4) Print and save graph
    scene_graph.pretty_print()
    scene_graph.save_json(graph_json_path)
    print(f"[INFO] Scene graph JSON saved to: {graph_json_path}")

    # 5) Visualize detections
    visualize_detections(
        image_path=image_path,
        objects=objects,
        output_path=vis_output_path
    )


if __name__ == "__main__":
    main()
