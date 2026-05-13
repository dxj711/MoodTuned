"""Smart Mood Engine for converting visual signals into music intent."""

from __future__ import annotations

import math
from typing import Any


class SmartMoodEngine:
    """Scores object, face, scene, color, OCR, and user intent signals."""

    def __init__(self) -> None:
        self.mood_keywords = {
            "happy": ["happy", "smile", "yellow", "friends", "celebrating", "graduation", "warm", "hangout"],
            "calm": ["calm", "blue", "beach", "nature", "minimal", "soft", "relaxing", "peaceful"],
            "energetic": ["gym", "workout", "party", "dance", "red", "vivid", "concert", "sports", "nightlife"],
            "romantic": ["couple", "romantic", "date", "sunset", "warm", "cozy", "food"],
            "emotional": ["rainy", "dark", "lonely", "moody", "cinematic", "neutral", "introspective"],
            "motivational": ["gym", "workout", "sports", "sharp", "fitness", "solo", "red"],
            "party": ["party", "nightlife", "dance", "crowd", "neon", "celebrating", "concert"],
            "nostalgic": ["memory", "friends", "warm", "graduation", "wedding", "sunset", "candid"],
            "lonely": ["solo", "rainy", "street", "dark", "blue", "introspective"],
            "cinematic": ["cinematic", "moody", "city", "night", "rainy", "contrast", "dark"],
            "friendship": ["friends", "group", "hangout", "shared", "smile", "memory"],
        }

        self.genre_map = {
            "happy": "Pop",
            "calm": "Lo-fi",
            "energetic": "Dance",
            "romantic": "R&B",
            "emotional": "Acoustic",
            "motivational": "Hip-Hop",
            "party": "EDM",
            "nostalgic": "Indie",
            "lonely": "Alternative",
            "cinematic": "Ambient",
            "friendship": "Pop",
        }

    def generate_mood_profile(
        self,
        features: dict[str, Any],
        preferred_language: str,
        preferred_genre: str,
        recommendation_style: str,
        mood_refinement: str = "",
    ) -> dict[str, Any]:
        scores = self._score_moods(features, mood_refinement)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        main_mood, main_score = ranked[0] if ranked else ("balanced", 50)

        mood_tags = self._build_mood_tags(ranked, features, mood_refinement)
        recommended_genre = preferred_genre if preferred_genre != "Auto" else self.genre_map.get(main_mood, "Pop")
        main_emotion = self._main_emotion(features, main_mood)
        playlist_title = self._playlist_title(mood_tags, preferred_language, recommendation_style)

        return {
            "main_mood": main_mood,
            "main_emotion": main_emotion,
            "mood_score": min(100, max(1, round(main_score))),
            "mood_tags": mood_tags,
            "vibe_description": self._vibe_description(features, mood_tags, main_emotion, recommended_genre),
            "recommended_genre": recommended_genre,
            "genre_query": self._genre_query(recommended_genre, main_mood, recommendation_style, preferred_language),
            "playlist_title": playlist_title,
            "playlist_variants": self._playlist_variants(main_mood, preferred_language),
            "score_breakdown": {k: round(v, 2) for k, v in ranked[:8]},
            "recommendation_style": recommendation_style,
            "preferred_language": preferred_language,
            "explanation_basis": self._basis(features),
        }

    def build_visual_mood_embedding(self, features: dict[str, Any], analysis: dict[str, Any]) -> list[float]:
        color = features.get("color_profile", {})
        faces = features.get("face_profile", {})
        moods = analysis.get("score_breakdown", {})
        mood_order = ["happy", "calm", "energetic", "romantic", "emotional", "party", "nostalgic", "cinematic"]

        vector = [
            float(color.get("brightness", 0.5)),
            float(color.get("saturation", 0.4)),
            float(color.get("contrast", 0.2)),
            min(float(faces.get("face_count", 0)) / 6.0, 1.0),
            float(faces.get("smile_ratio", 0)),
        ]
        vector.extend(float(moods.get(mood, 0)) / 100.0 for mood in mood_order)
        return vector

    def _score_moods(self, features: dict[str, Any], mood_refinement: str) -> dict[str, float]:
        tags = set(features.get("contextual_tags", []))
        labels = " ".join(label.get("label", "") for label in features.get("scene_labels", []))
        ocr = features.get("ocr_text", "")
        aesthetic = features.get("aesthetic_style", "")
        social = features.get("social_context", "")
        activities = " ".join(features.get("activities", []))
        face = features.get("face_profile", {})
        colors = features.get("color_profile", {})
        psychology = " ".join(item.get("name", "") + " " + item.get("emotion", "") for item in colors.get("psychology", []))

        corpus = f"{labels} {ocr} {aesthetic} {social} {activities} {psychology} {mood_refinement}".lower()

        scores: dict[str, float] = {}
        for mood, keywords in self.mood_keywords.items():
            keyword_hits = sum(1 for keyword in keywords if keyword in corpus or keyword in tags)
            scores[mood] = 20 + keyword_hits * 10

        brightness = float(colors.get("brightness", 0.5))
        saturation = float(colors.get("saturation", 0.4))
        contrast = float(colors.get("contrast", 0.2))
        smile_ratio = float(face.get("smile_ratio", 0))
        face_count = int(face.get("face_count", 0))
        face_emotion = str(face.get("dominant_emotion", "")).lower()

        scores["happy"] += smile_ratio * 25 + max(brightness - 0.45, 0) * 22
        scores["friendship"] += min(face_count, 5) * 7 + smile_ratio * 12
        scores["party"] += saturation * 20 + (12 if face_count >= 4 else 0)
        scores["energetic"] += saturation * 18 + contrast * 14
        scores["calm"] += max(0.0, 0.7 - saturation) * 18 + max(brightness - 0.4, 0) * 8
        scores["emotional"] += max(0.45 - brightness, 0) * 35 + contrast * 10
        scores["cinematic"] += max(0.48 - brightness, 0) * 20 + contrast * 18
        scores["lonely"] += (10 if face_count <= 1 else 0) + max(0.43 - brightness, 0) * 20
        scores["motivational"] += contrast * 12 + (10 if "workout" in corpus or "gym" in corpus else 0)
        scores["romantic"] += (12 if "couple" in corpus else 0) + (8 if "sunset" in corpus else 0)
        scores["nostalgic"] += (8 if "warm" in corpus else 0) + (8 if "candid" in corpus else 0)

        if face_emotion in {"happy", "surprise"}:
            scores["happy"] += 18
            scores["party"] += 8
        elif face_emotion in {"sad", "fear", "angry"}:
            scores["emotional"] += 20
            scores["cinematic"] += 6

        return {key: min(100.0, value) for key, value in scores.items()}

    def _build_mood_tags(
        self,
        ranked: list[tuple[str, float]],
        features: dict[str, Any],
        mood_refinement: str,
    ) -> list[str]:
        tags = [mood for mood, score in ranked[:5] if score >= 35]
        tags.extend(features.get("activities", [])[:3])
        tags.extend(features.get("aesthetic_style", "").split()[:2])
        tags.extend(features.get("social_context", "").replace(" or ", " ").replace(" and ", " ").split()[:3])
        if mood_refinement:
            tags.extend(mood_refinement.lower().split()[:3])
        unique: list[str] = []
        for tag in tags:
            tag = tag.strip().lower()
            if len(tag) > 2 and tag not in unique:
                unique.append(tag)
        return unique[:12]

    def _main_emotion(self, features: dict[str, Any], main_mood: str) -> str:
        face_emotion = features.get("face_profile", {}).get("dominant_emotion")
        if face_emotion and face_emotion != "neutral":
            return str(face_emotion).title()
        mapping = {
            "happy": "Joy",
            "calm": "Peace",
            "energetic": "Excitement",
            "romantic": "Affection",
            "emotional": "Melancholy",
            "motivational": "Drive",
            "party": "Celebration",
            "nostalgic": "Nostalgia",
            "lonely": "Loneliness",
            "cinematic": "Reflection",
            "friendship": "Belonging",
        }
        return mapping.get(main_mood, "Balanced")

    def _vibe_description(
        self,
        features: dict[str, Any],
        mood_tags: list[str],
        main_emotion: str,
        genre: str,
    ) -> str:
        colors = features.get("color_profile", {})
        faces = features.get("face_profile", {})
        caption = features.get("caption", "The image")
        tag_text = ", ".join(mood_tags[:5])
        return (
            f"{caption} The mood engine reads the moment as {main_emotion.lower()} with {tag_text} cues. "
            f"Brightness is {colors.get('brightness_label', 'balanced')}, saturation is {colors.get('saturation_label', 'natural')}, "
            f"and face analysis suggests {faces.get('group_energy', 'scene-led energy')}. "
            f"{genre} is a strong fit for this image."
        )

    def _genre_query(self, genre: str, main_mood: str, style: str, language: str) -> str:
        style_terms = {
            "Trending": "popular trending",
            "Classic": "classic timeless",
            "Indie": "indie alternative",
            "Lo-fi": "lofi chill",
            "Viral": "viral reels",
        }
        language_terms = {
            "English": "english",
            "Hindi": "hindi bollywood",
            "Malayalam": "malayalam",
            "Tamil": "tamil",
            "Korean": "korean kpop",
        }
        return f"{language_terms.get(language, language)} {style_terms.get(style, style)} {main_mood} {genre}"

    def _playlist_title(self, mood_tags: list[str], language: str, style: str) -> str:
        lead = " ".join(tag.title() for tag in mood_tags[:2]) or "Visual Mood"
        return f"{lead} {style} Mix ({language})"

    def _playlist_variants(self, main_mood: str, language: str) -> dict[str, str]:
        return {
            "Top 5 songs for this vibe": f"{language} {main_mood} top songs",
            "Chill playlist": f"{language} chill calm lo-fi",
            "Party playlist": f"{language} party dance energetic",
            "Emotional playlist": f"{language} emotional acoustic cinematic",
        }

    def _basis(self, features: dict[str, Any]) -> str:
        face = features.get("face_profile", {})
        color = features.get("color_profile", {})
        scene = features.get("scene_understanding", "")
        return (
            f"{scene} Face count: {face.get('face_count', 0)}, smiles: {face.get('smile_count', 0)}. "
            f"Dominant palette is {color.get('temperature', 'neutral')} with "
            f"{color.get('brightness_label', 'balanced')} brightness and {color.get('saturation_label', 'natural')} saturation."
        )
