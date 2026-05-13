"""Image understanding layer for MoodTune AI.

The module combines optional AI models with deterministic computer-vision
signals so the app remains useful even before large model weights are present.
"""

from __future__ import annotations

import importlib.util
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageStat

try:
    import cv2
except Exception:  # pragma: no cover - OpenCV is optional on Streamlit Cloud.
    cv2 = None


@dataclass(frozen=True)
class LabelScore:
    label: str
    score: float


class ImageAnalyzer:
    """Extracts caption, scene, color, face, OCR, and activity signals."""

    def __init__(self) -> None:
        self._captioner = None
        self._zero_shot = None
        self._deepface = None
        self._load_errors: list[str] = []

        self.scene_labels = [
            "friends hanging out",
            "beach sunset",
            "gym workout selfie",
            "rainy street",
            "party nightlife",
            "romantic couple",
            "solo portrait",
            "concert crowd",
            "mountain travel",
            "city night",
            "cafe hangout",
            "wedding celebration",
            "graduation memory",
            "office work",
            "food date",
            "nature landscape",
            "sports activity",
            "car road trip",
        ]

    def analyze(self, image: Image.Image) -> dict[str, Any]:
        image = image.convert("RGB")
        np_image = np.array(image)

        color_profile = self._analyze_colors(image)
        face_profile = self._analyze_faces(np_image)
        ai_labels = self._classify_scene(image)
        caption = self._caption_image(image, ai_labels, color_profile, face_profile)
        ocr_text = self._extract_text(np_image)

        activities = self._infer_activities(ai_labels, face_profile, ocr_text)
        social_context = self._infer_social_context(face_profile, ai_labels)
        aesthetic = self._infer_aesthetic(color_profile, ai_labels)
        contextual_tags = self._build_contextual_tags(
            labels=ai_labels,
            colors=color_profile,
            faces=face_profile,
            activities=activities,
            ocr_text=ocr_text,
            aesthetic=aesthetic,
            social_context=social_context,
        )

        scene = self._scene_sentence(ai_labels, social_context, activities, aesthetic)
        emotion = self._emotion_sentence(color_profile, face_profile, ai_labels, contextual_tags)

        return {
            "caption": caption,
            "scene_labels": [label.__dict__ for label in ai_labels],
            "scene_understanding": scene,
            "emotional_interpretation": emotion,
            "contextual_tags": contextual_tags,
            "aesthetic_style": aesthetic,
            "social_context": social_context,
            "activities": activities,
            "color_profile": color_profile,
            "color_psychology": color_profile["psychology"],
            "face_profile": face_profile,
            "ocr_text": ocr_text,
            "model_notes": self._load_errors[-5:],
        }

    def _caption_image(
        self,
        image: Image.Image,
        labels: list[LabelScore],
        colors: dict[str, Any],
        faces: dict[str, Any],
    ) -> str:
        caption = self._try_blip_caption(image)
        if caption:
            return caption

        top_label = labels[0].label if labels else "a photographed moment"
        brightness = colors.get("brightness_label", "balanced")
        energy = faces.get("group_energy", "quiet")
        return f"A {brightness} image showing {top_label}, with a {energy} social energy and a {colors.get('temperature', 'neutral')} visual tone."

    def _try_blip_caption(self, image: Image.Image) -> str | None:
        if self._captioner is False:
            return None
        try:
            if self._captioner is None:
                if importlib.util.find_spec("transformers") is None:
                    self._captioner = False
                    self._load_errors.append("Transformers is not installed; using heuristic captioning.")
                    return None
                from transformers import pipeline

                self._captioner = pipeline(
                    "image-to-text",
                    model="Salesforce/blip-image-captioning-base",
                )
            result = self._captioner(image, max_new_tokens=40)
            if result and isinstance(result, list):
                return str(result[0].get("generated_text", "")).strip()
        except Exception as exc:  # pragma: no cover - model availability varies by machine.
            self._captioner = False
            self._load_errors.append(f"BLIP captioning unavailable: {exc}")
        return None

    def _classify_scene(self, image: Image.Image) -> list[LabelScore]:
        labels = self._try_clip_labels(image)
        if labels:
            return labels
        return self._heuristic_scene_labels(image)

    def _try_clip_labels(self, image: Image.Image) -> list[LabelScore]:
        if self._zero_shot is False:
            return []
        try:
            if self._zero_shot is None:
                if importlib.util.find_spec("transformers") is None:
                    self._zero_shot = False
                    return []
                from transformers import pipeline

                self._zero_shot = pipeline(
                    "zero-shot-image-classification",
                    model="openai/clip-vit-base-patch32",
                )
            results = self._zero_shot(image, candidate_labels=self.scene_labels)
            return [
                LabelScore(label=item["label"], score=round(float(item["score"]), 4))
                for item in results[:6]
            ]
        except Exception as exc:  # pragma: no cover - model availability varies by machine.
            self._zero_shot = False
            self._load_errors.append(f"CLIP scene classification unavailable: {exc}")
            return []

    def _heuristic_scene_labels(self, image: Image.Image) -> list[LabelScore]:
        arr = np.array(image.resize((128, 128))).astype(np.float32) / 255.0
        if cv2 is not None:
            hsv = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
            hue = hsv[:, :, 0]
            saturation = hsv[:, :, 1].mean() / 255.0
            value = hsv[:, :, 2].mean() / 255.0
            blue = np.mean((hue > 90) & (hue < 130))
            orange = np.mean((hue > 8) & (hue < 30))
            green = np.mean((hue > 35) & (hue < 85))
            dark = np.mean(hsv[:, :, 2] < 70)
        else:
            red, green_channel, blue_channel = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            value = float(np.max(arr, axis=2).mean())
            saturation = float((np.max(arr, axis=2) - np.min(arr, axis=2)).mean())
            blue = float(np.mean((blue_channel > red * 1.12) & (blue_channel > green_channel * 1.05)))
            orange = float(np.mean((red > 0.48) & (green_channel > 0.25) & (blue_channel < 0.32)))
            green = float(np.mean((green_channel > red * 1.05) & (green_channel > blue_channel * 1.05)))
            dark = float(np.mean(value < 0.28))
        edge_density = self._edge_density(arr)

        scores: dict[str, float] = {
            "nature landscape": green * 0.75 + value * 0.2,
            "beach sunset": blue * 0.35 + orange * 0.45 + saturation * 0.2,
            "city night": dark * 0.5 + edge_density * 0.25 + saturation * 0.1,
            "rainy street": dark * 0.35 + blue * 0.25 + edge_density * 0.2,
            "party nightlife": dark * 0.3 + saturation * 0.45 + edge_density * 0.15,
            "solo portrait": 0.2 + value * 0.1,
            "friends hanging out": 0.2 + saturation * 0.2 + value * 0.2,
        }
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        total = sum(score for _, score in ranked[:6]) or 1.0
        return [LabelScore(label=label, score=round(score / total, 4)) for label, score in ranked[:6]]

    def _edge_density(self, arr: np.ndarray) -> float:
        gray = self._rgb_to_gray_float(arr)
        if cv2 is not None:
            edges = cv2.Canny((gray * 255).astype(np.uint8), 80, 160)
            return float(np.mean(edges > 0))
        gy, gx = np.gradient(gray)
        return float(np.mean(np.sqrt(gx * gx + gy * gy) > 0.12))

    def _analyze_colors(self, image: Image.Image) -> dict[str, Any]:
        small = image.resize((140, 140))
        arr = np.array(small)

        stat = ImageStat.Stat(small)
        brightness = float(np.mean(stat.mean) / 255.0)
        saturation = self._mean_saturation(arr)
        contrast = float(np.std(self._rgb_to_gray_float(arr.astype(np.float32) / 255.0)))

        palette = self._dominant_palette(arr, k=5)
        psychology = [self._color_psychology(item["rgb"], item["hex"]) for item in palette]
        temperature = self._temperature(psychology)

        return {
            "brightness": round(brightness, 3),
            "saturation": round(saturation, 3),
            "contrast": round(contrast, 3),
            "brightness_label": self._brightness_label(brightness),
            "saturation_label": self._saturation_label(saturation),
            "temperature": temperature,
            "palette": palette,
            "psychology": psychology,
        }

    def _dominant_palette(self, arr: np.ndarray, k: int = 5) -> list[dict[str, Any]]:
        pixels = arr.reshape((-1, 3)).astype(np.float32)
        if cv2 is not None:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.2)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            counts = Counter(labels.flatten())
            total = len(labels)
            ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
            palette = []
            for idx, count in ranked:
                rgb = [int(c) for c in centers[idx]]
                palette.append(
                    {
                        "rgb": rgb,
                        "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                        "percentage": round(count / total, 3),
                    }
                )
            return palette

        quantized = (pixels // 32 * 32).astype(int)
        counts = Counter(map(tuple, quantized))
        total = len(quantized)
        palette = []
        for rgb_tuple, count in counts.most_common(k):
            rgb = [min(int(channel + 16), 255) for channel in rgb_tuple]
            palette.append(
                {
                    "rgb": rgb,
                    "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
                    "percentage": round(count / total, 3),
                }
            )
        return palette

    def _color_psychology(self, rgb: list[int], hex_value: str) -> dict[str, Any]:
        r, g, b = rgb
        hue, sat, val = self._rgb_to_hsv_degrees(r, g, b)

        if val < 65:
            name, emotion = "Dark tones", "emotional, mysterious, cinematic"
        elif sat < 35 and val > 185:
            name, emotion = "Light neutrals", "clean, soft, minimal"
        elif 90 <= hue <= 130:
            name, emotion = "Blue", "calm, reflective, peaceful"
        elif hue <= 8 or hue >= 170:
            name, emotion = "Red", "energetic, passionate, intense"
        elif 18 <= hue <= 35:
            name, emotion = "Yellow", "happy, warm, optimistic"
        elif 36 <= hue <= 85:
            name, emotion = "Green", "fresh, natural, grounded"
        elif 131 <= hue <= 160:
            name, emotion = "Purple", "dreamy, stylish, introspective"
        else:
            name, emotion = "Warm tones", "cozy, social, nostalgic"
        return {"name": name, "emotion": emotion, "hex": hex_value, "rgb": rgb}

    def _temperature(self, psychology: list[dict[str, Any]]) -> str:
        labels = [item["name"] for item in psychology]
        warm = sum(label in {"Red", "Yellow", "Warm tones"} for label in labels)
        cool = sum(label in {"Blue", "Green", "Purple"} for label in labels)
        if warm > cool:
            return "warm"
        if cool > warm:
            return "cool"
        return "neutral"

    def _brightness_label(self, brightness: float) -> str:
        if brightness >= 0.72:
            return "bright"
        if brightness <= 0.33:
            return "dark"
        return "balanced"

    def _saturation_label(self, saturation: float) -> str:
        if saturation >= 0.55:
            return "vivid"
        if saturation <= 0.22:
            return "muted"
        return "natural"

    def _analyze_faces(self, np_image: np.ndarray) -> dict[str, Any]:
        if cv2 is None:
            return {
                "face_count": 0,
                "smile_count": 0,
                "smile_ratio": 0,
                "dominant_emotion": "neutral",
                "group_energy": "scene-led energy",
                "face_boxes": [],
            }

        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(35, 35))

        smiles = 0
        face_boxes: list[dict[str, int]] = []
        for x, y, w, h in faces:
            face_boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
            roi_gray = gray[y : y + h, x : x + w]
            detected_smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=18)
            if len(detected_smiles) > 0:
                smiles += 1

        emotion = self._try_deepface_emotion(np_image)
        if not emotion:
            if len(faces) and smiles / max(len(faces), 1) >= 0.45:
                emotion = "happy"
            elif len(faces) >= 4:
                emotion = "excited"
            else:
                emotion = "neutral"

        if len(faces) >= 4:
            group_energy = "crowd excitement"
        elif len(faces) >= 2 and smiles:
            group_energy = "friendly and happy"
        elif len(faces) == 1 and smiles:
            group_energy = "positive solo energy"
        elif len(faces) == 1:
            group_energy = "introspective solo energy"
        else:
            group_energy = "scene-led energy"

        return {
            "face_count": int(len(faces)),
            "smile_count": int(smiles),
            "smile_ratio": round(smiles / max(len(faces), 1), 3),
            "dominant_emotion": emotion,
            "group_energy": group_energy,
            "face_boxes": face_boxes,
        }

    def _try_deepface_emotion(self, np_image: np.ndarray) -> str | None:
        if self._deepface is False:
            return None
        try:
            if self._deepface is None:
                if importlib.util.find_spec("deepface") is None:
                    self._deepface = False
                    return None
                from deepface import DeepFace

                self._deepface = DeepFace
            result = self._deepface.analyze(np_image, actions=["emotion"], enforce_detection=False)
            if isinstance(result, list) and result:
                return str(result[0].get("dominant_emotion", "")).lower() or None
            if isinstance(result, dict):
                return str(result.get("dominant_emotion", "")).lower() or None
        except Exception as exc:  # pragma: no cover - dependency/model availability varies.
            self._deepface = False
            self._load_errors.append(f"DeepFace emotion detection unavailable: {exc}")
        return None

    def _extract_text(self, np_image: np.ndarray) -> str:
        try:
            if importlib.util.find_spec("pytesseract") is None:
                return ""
            import pytesseract

            text = pytesseract.image_to_string(np_image)
            return " ".join(text.split())[:500]
        except Exception:
            return ""

    def _rgb_to_gray_float(self, arr: np.ndarray) -> np.ndarray:
        if arr.max() > 1.0:
            arr = arr.astype(np.float32) / 255.0
        return arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114

    def _mean_saturation(self, arr: np.ndarray) -> float:
        normalized = arr.astype(np.float32) / 255.0
        max_channel = np.max(normalized, axis=2)
        min_channel = np.min(normalized, axis=2)
        return float(np.mean(max_channel - min_channel))

    def _rgb_to_hsv_degrees(self, r: int, g: int, b: int) -> tuple[int, int, int]:
        rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
        max_c = max(rf, gf, bf)
        min_c = min(rf, gf, bf)
        delta = max_c - min_c
        if delta == 0:
            hue = 0
        elif max_c == rf:
            hue = (60 * ((gf - bf) / delta) + 360) % 360
        elif max_c == gf:
            hue = 60 * ((bf - rf) / delta + 2)
        else:
            hue = 60 * ((rf - gf) / delta + 4)
        saturation = 0 if max_c == 0 else delta / max_c
        return int(hue / 2), int(saturation * 255), int(max_c * 255)

    def _infer_activities(
        self,
        labels: list[LabelScore],
        faces: dict[str, Any],
        ocr_text: str,
    ) -> list[str]:
        label_text = " ".join(label.label for label in labels)
        text = f"{label_text} {ocr_text}".lower()
        rules = {
            "workout": ["gym", "workout", "sports", "fitness"],
            "dancing": ["party", "nightlife", "concert", "dance"],
            "traveling": ["travel", "road trip", "mountain", "beach", "city"],
            "relaxing": ["beach", "cafe", "nature", "sunset"],
            "celebrating": ["wedding", "graduation", "party", "celebration"],
            "hanging out": ["friends", "cafe", "food", "hangout"],
            "romantic moment": ["romantic", "couple", "sunset", "date"],
            "working": ["office", "work", "laptop"],
        }
        found = [activity for activity, terms in rules.items() if any(term in text for term in terms)]
        if faces.get("face_count", 0) >= 3 and "hanging out" not in found:
            found.append("hanging out")
        return found[:5] or ["visual storytelling"]

    def _infer_social_context(self, faces: dict[str, Any], labels: list[LabelScore]) -> str:
        label_text = " ".join(label.label for label in labels).lower()
        face_count = faces.get("face_count", 0)
        if "couple" in label_text or face_count == 2:
            return "couple or close relationship"
        if "friends" in label_text or face_count >= 3:
            return "friend group or shared memory"
        if face_count == 1:
            return "solo self-expression"
        if "party" in label_text or "concert" in label_text:
            return "crowd or nightlife setting"
        return "environment-focused scene"

    def _infer_aesthetic(self, colors: dict[str, Any], labels: list[LabelScore]) -> str:
        label_text = " ".join(label.label for label in labels).lower()
        brightness = colors["brightness"]
        saturation = colors["saturation"]
        contrast = colors["contrast"]
        temperature = colors["temperature"]
        if "rainy" in label_text or (brightness < 0.36 and contrast > 0.22):
            return "cinematic moody"
        if "party" in label_text or (saturation > 0.58 and brightness < 0.48):
            return "neon nightlife"
        if "sunset" in label_text or temperature == "warm" and saturation > 0.34:
            return "warm nostalgic"
        if brightness > 0.72 and saturation < 0.28:
            return "soft minimal"
        if "gym" in label_text or contrast > 0.3:
            return "high-energy sharp"
        return "natural candid"

    def _build_contextual_tags(
        self,
        labels: list[LabelScore],
        colors: dict[str, Any],
        faces: dict[str, Any],
        activities: list[str],
        ocr_text: str,
        aesthetic: str,
        social_context: str,
    ) -> list[str]:
        tags: list[str] = []
        for label in labels[:5]:
            tags.extend(label.label.split())
        tags.extend(activities)
        tags.extend(aesthetic.split())
        tags.extend(social_context.replace(" or ", " ").replace(" and ", " ").split())
        tags.extend(colors["temperature"].split())
        tags.extend(item["name"].lower().replace(" tones", "").split()[0] for item in colors["psychology"][:3])
        if faces.get("face_count", 0) > 0:
            tags.append(f"{faces['face_count']}-face")
            tags.append(faces.get("dominant_emotion", "neutral"))
        if faces.get("smile_ratio", 0) > 0.4:
            tags.extend(["smile", "happy"])
        if ocr_text:
            tags.extend(["text-in-image", "captioned"])

        cleaned = []
        for tag in tags:
            normalized = tag.lower().strip(" ,.-_/")
            if len(normalized) > 2 and normalized not in cleaned:
                cleaned.append(normalized)
        return cleaned[:22]

    def _scene_sentence(
        self,
        labels: list[LabelScore],
        social_context: str,
        activities: list[str],
        aesthetic: str,
    ) -> str:
        top = labels[0].label if labels else "an expressive scene"
        activity_text = ", ".join(activities[:3])
        return (
            f"The image reads as {top}, with {social_context}. "
            f"The likely activity is {activity_text}, presented in a {aesthetic} aesthetic."
        )

    def _emotion_sentence(
        self,
        colors: dict[str, Any],
        faces: dict[str, Any],
        labels: list[LabelScore],
        tags: list[str],
    ) -> str:
        face_emotion = faces.get("dominant_emotion", "neutral")
        color_emotion = colors["psychology"][0]["emotion"] if colors.get("psychology") else "balanced"
        label_text = labels[0].label if labels else "visual moment"
        social = faces.get("group_energy", "scene-led energy")
        keywords = ", ".join(tags[:5])
        return (
            f"The strongest cues suggest {face_emotion} emotion and {social}. "
            f"The dominant color psychology feels {color_emotion}. "
            f"Together, the {label_text} cues point toward {keywords}."
        )
