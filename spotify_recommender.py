"""Spotify recommendation layer for MoodTune AI."""

from __future__ import annotations

import os
import random
from typing import Any
from urllib.parse import quote_plus

try:
    import streamlit as st
except Exception:  # pragma: no cover - streamlit is present at app runtime.
    st = None

try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
except Exception:  # pragma: no cover - dependency may be installed later.
    spotipy = None
    SpotifyClientCredentials = None
    SpotifyOAuth = None

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


class SpotifyRecommender:
    """Searches Spotify dynamically and formats explainable recommendations."""

    def __init__(self) -> None:
        if load_dotenv:
            load_dotenv()

        self.client_id = self._secret("SPOTIFY_CLIENT_ID")
        self.client_secret = self._secret("SPOTIFY_CLIENT_SECRET")
        self.redirect_uri = self._secret("SPOTIFY_REDIRECT_URI", "http://localhost:8501")
        self.market = self._secret("SPOTIFY_MARKET", "IN")
        self.sp = self._client_credentials_client()
        self.user_sp = None

    @property
    def is_configured(self) -> bool:
        return bool(self.sp)

    @property
    def can_export_playlist(self) -> bool:
        return bool(self.client_id and self.client_secret and spotipy and SpotifyOAuth)

    def recommend_songs(
        self,
        mood_profile: dict[str, Any],
        preferred_language: str,
        preferred_genre: str,
        recommendation_style: str,
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        if self.sp:
            songs = self._spotify_search(mood_profile, preferred_language, preferred_genre, recommendation_style, limit)
            if songs:
                return songs
        return self._fallback_songs(mood_profile, preferred_language, preferred_genre, recommendation_style, limit)

    def search_playlists(
        self,
        mood_profile: dict[str, Any],
        preferred_language: str,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        query = f"{preferred_language} {mood_profile.get('main_mood', 'mood')} {mood_profile.get('recommended_genre', 'playlist')}"
        if self.sp:
            try:
                data = self.sp.search(q=query, type="playlist", limit=limit, market=self.market)
                playlists = []
                for item in data.get("playlists", {}).get("items", []) or []:
                    images = item.get("images") or []
                    owner = item.get("owner", {}).get("display_name", "Spotify")
                    playlists.append(
                        {
                            "name": item.get("name", "Mood playlist"),
                            "url": item.get("external_urls", {}).get("spotify", "https://open.spotify.com/search"),
                            "image": images[0]["url"] if images else None,
                            "owner": owner,
                        }
                    )
                if playlists:
                    return playlists
            except Exception:
                pass

        return [
            {
                "name": title,
                "url": f"https://open.spotify.com/search/{quote_plus(query_text)}",
                "image": "https://placehold.co/640x360/111827/f8fafc?text=MoodTune+Playlist",
                "owner": "MoodTune AI",
            }
            for title, query_text in mood_profile.get("playlist_variants", {}).items()
        ][:limit]

    def create_playlist(
        self,
        title: str,
        description: str,
        track_uris: list[str],
        public: bool = False,
    ) -> dict[str, Any]:
        if not self.can_export_playlist:
            return {"ok": False, "error": "Spotify OAuth credentials are not configured."}
        if not track_uris:
            return {"ok": False, "error": "No Spotify track URIs are available for export."}
        try:
            if not self.user_sp:
                auth = SpotifyOAuth(
                    client_id=self.client_id,
                    client_secret=self.client_secret,
                    redirect_uri=self.redirect_uri,
                    scope="playlist-modify-private playlist-modify-public",
                    open_browser=True,
                )
                self.user_sp = spotipy.Spotify(auth_manager=auth)
            user = self.user_sp.current_user()
            playlist = self.user_sp.user_playlist_create(
                user=user["id"],
                name=title,
                public=public,
                description=description[:280],
            )
            self.user_sp.playlist_add_items(playlist["id"], track_uris[:100])
            return {"ok": True, "url": playlist.get("external_urls", {}).get("spotify")}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _client_credentials_client(self) -> Any | None:
        if not (spotipy and SpotifyClientCredentials and self.client_id and self.client_secret):
            return None
        try:
            auth = SpotifyClientCredentials(client_id=self.client_id, client_secret=self.client_secret)
            return spotipy.Spotify(auth_manager=auth, requests_timeout=12, retries=2)
        except Exception:
            return None

    def _secret(self, key: str, default: str = "") -> str:
        value = os.getenv(key)
        if value:
            return value
        if st is not None:
            try:
                return str(st.secrets.get(key, default))
            except Exception:
                return default
        return default

    def _spotify_search(
        self,
        mood_profile: dict[str, Any],
        preferred_language: str,
        preferred_genre: str,
        recommendation_style: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        query = mood_profile.get("genre_query") or self._query(mood_profile, preferred_language, preferred_genre, recommendation_style)
        songs: list[dict[str, Any]] = []
        seen: set[str] = set()

        query_variants = [
            query,
            f"{preferred_language} {mood_profile.get('main_mood', '')} {preferred_genre}",
            f"{recommendation_style} {mood_profile.get('recommended_genre', '')} songs",
            f"{' '.join(mood_profile.get('mood_tags', [])[:4])} songs",
        ]

        for q in query_variants:
            if len(songs) >= limit:
                break
            try:
                data = self.sp.search(q=q, type="track", limit=limit, market=self.market)
            except Exception:
                continue
            for track in data.get("tracks", {}).get("items", []) or []:
                track_id = track.get("id")
                if not track_id or track_id in seen:
                    continue
                seen.add(track_id)
                artists = ", ".join(artist.get("name", "") for artist in track.get("artists", [])) or "Unknown artist"
                album = track.get("album", {})
                images = album.get("images") or []
                songs.append(
                    {
                        "name": track.get("name", "Unknown song"),
                        "artist": artists,
                        "album": album.get("name", ""),
                        "album_cover": images[0]["url"] if images else None,
                        "spotify_url": track.get("external_urls", {}).get("spotify"),
                        "spotify_uri": track.get("uri"),
                        "spotify_id": track_id,
                        "preview_url": track.get("preview_url"),
                        "popularity": track.get("popularity", 0),
                        "genre_match": self._match_percentage(track, mood_profile, recommendation_style),
                        "why": self._why_song_matches(track, mood_profile),
                    }
                )
                if len(songs) >= limit:
                    break
        return sorted(songs, key=lambda item: item.get("genre_match", 0), reverse=True)[:limit]

    def _query(
        self,
        mood_profile: dict[str, Any],
        preferred_language: str,
        preferred_genre: str,
        style: str,
    ) -> str:
        genre = preferred_genre if preferred_genre != "Auto" else mood_profile.get("recommended_genre", "Pop")
        tags = " ".join(mood_profile.get("mood_tags", [])[:4])
        return f"{preferred_language} {style} {genre} {tags}"

    def _match_percentage(self, track: dict[str, Any], mood_profile: dict[str, Any], style: str) -> int:
        name = track.get("name", "").lower()
        album = track.get("album", {}).get("name", "").lower()
        tags = mood_profile.get("mood_tags", [])
        score = 55
        score += min(int(track.get("popularity", 0)) // 6, 15)
        score += sum(4 for tag in tags if tag.lower() in name or tag.lower() in album)
        if style == "Trending":
            score += min(int(track.get("popularity", 0)) // 10, 10)
        if style in {"Indie", "Lo-fi"} and any(style.lower().replace("-", "") in text for text in [name, album]):
            score += 8
        return min(score, 98)

    def _why_song_matches(self, track: dict[str, Any], mood_profile: dict[str, Any]) -> str:
        artist = ", ".join(artist.get("name", "") for artist in track.get("artists", [])) or "the artist"
        tags = ", ".join(mood_profile.get("mood_tags", [])[:4])
        basis = mood_profile.get("explanation_basis", "")
        return (
            f"{track.get('name', 'This song')} by {artist} fits because the image was scored for {tags}. "
            f"{basis}"
        )

    def _fallback_songs(
        self,
        mood_profile: dict[str, Any],
        preferred_language: str,
        preferred_genre: str,
        style: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        catalog = self._fallback_catalog().get(preferred_language, self._fallback_catalog()["English"])
        main = mood_profile.get("main_mood", "happy")
        genre = preferred_genre if preferred_genre != "Auto" else mood_profile.get("recommended_genre", "Pop")
        ranked = []
        for song in catalog:
            text = " ".join(song.get("tags", []) + [song.get("genre", "")]).lower()
            score = 62
            if main in text:
                score += 18
            if genre.lower() in text:
                score += 10
            if style.lower().replace("-", "") in text.replace("-", ""):
                score += 8
            score += random.randint(0, 5)
            ranked.append((score, song))
        ranked.sort(key=lambda item: item[0], reverse=True)

        results = []
        for score, song in ranked[:limit]:
            search = f"{song['name']} {song['artist']}"
            results.append(
                {
                    "name": song["name"],
                    "artist": song["artist"],
                    "album": "",
                    "album_cover": f"https://placehold.co/420x420/111827/f8fafc?text={quote_plus(song['name'][:16])}",
                    "spotify_url": f"https://open.spotify.com/search/{quote_plus(search)}",
                    "search_url": f"https://open.spotify.com/search/{quote_plus(search)}",
                    "spotify_uri": None,
                    "spotify_id": None,
                    "preview_url": None,
                    "popularity": 0,
                    "genre_match": min(score, 96),
                    "why": self._fallback_reason(song, mood_profile),
                }
            )
        return results

    def _fallback_reason(self, song: dict[str, Any], mood_profile: dict[str, Any]) -> str:
        tags = ", ".join(mood_profile.get("mood_tags", [])[:4])
        return (
            f"{song['name']} matches the detected {tags} mood and the requested "
            f"{mood_profile.get('preferred_language', 'language')} listening context."
        )

    def _fallback_catalog(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "English": [
                {"name": "Good as Hell", "artist": "Lizzo", "genre": "Pop", "tags": ["happy", "energetic", "viral"]},
                {"name": "As It Was", "artist": "Harry Styles", "genre": "Pop", "tags": ["nostalgic", "trending"]},
                {"name": "Blinding Lights", "artist": "The Weeknd", "genre": "Synth Pop", "tags": ["party", "nightlife", "energetic"]},
                {"name": "Sunflower", "artist": "Post Malone, Swae Lee", "genre": "Pop", "tags": ["friendship", "happy", "chill"]},
                {"name": "Someone Like You", "artist": "Adele", "genre": "Ballad", "tags": ["emotional", "lonely", "classic"]},
                {"name": "Weightless", "artist": "Marconi Union", "genre": "Ambient", "tags": ["calm", "lofi", "peaceful"]},
                {"name": "Eye of the Tiger", "artist": "Survivor", "genre": "Rock", "tags": ["motivational", "workout", "classic"]},
                {"name": "Electric Feel", "artist": "MGMT", "genre": "Indie", "tags": ["indie", "party", "warm"]},
            ],
            "Hindi": [
                {"name": "Ilahi", "artist": "Arijit Singh", "genre": "Bollywood", "tags": ["travel", "happy", "friendship"]},
                {"name": "Badtameez Dil", "artist": "Benny Dayal", "genre": "Bollywood Dance", "tags": ["party", "energetic"]},
                {"name": "Channa Mereya", "artist": "Arijit Singh", "genre": "Bollywood", "tags": ["emotional", "classic"]},
                {"name": "Kesariya", "artist": "Arijit Singh", "genre": "Romantic", "tags": ["romantic", "warm"]},
                {"name": "Apna Time Aayega", "artist": "Ranveer Singh, Dub Sharma", "genre": "Hip-Hop", "tags": ["motivational", "workout"]},
                {"name": "Khaabon Ke Parinday", "artist": "Alyssa Mendonsa, Mohit Chauhan", "genre": "Bollywood", "tags": ["calm", "travel"]},
            ],
            "Malayalam": [
                {"name": "Malare", "artist": "Vijay Yesudas", "genre": "Romantic", "tags": ["romantic", "emotional"]},
                {"name": "Aaradhike", "artist": "Sooraj Santhosh, Madhuvanthi Narayan", "genre": "Melody", "tags": ["calm", "romantic"]},
                {"name": "Jeevamshamayi", "artist": "Harishankar K.S., Shreya Ghoshal", "genre": "Melody", "tags": ["emotional", "romantic"]},
                {"name": "Parudeesa", "artist": "Sreenath Bhasi", "genre": "Indie", "tags": ["indie", "nostalgic"]},
                {"name": "Pavizha Mazha", "artist": "K.S. Harisankar", "genre": "Melody", "tags": ["calm", "emotional"]},
            ],
            "Tamil": [
                {"name": "Vaathi Coming", "artist": "Anirudh Ravichander, Gana Balachandar", "genre": "Dance", "tags": ["party", "energetic"]},
                {"name": "Munbe Vaa", "artist": "Naresh Iyer, Shreya Ghoshal", "genre": "Romantic", "tags": ["romantic", "classic"]},
                {"name": "Enjoy Enjaami", "artist": "Dhee, Arivu", "genre": "Indie Pop", "tags": ["indie", "happy"]},
                {"name": "Maruvaarthai", "artist": "Sid Sriram", "genre": "Melody", "tags": ["emotional", "romantic"]},
                {"name": "Aalaporaan Thamizhan", "artist": "A.R. Rahman", "genre": "Motivational", "tags": ["motivational", "energetic"]},
            ],
            "Korean": [
                {"name": "Dynamite", "artist": "BTS", "genre": "K-pop", "tags": ["happy", "party", "viral"]},
                {"name": "Ditto", "artist": "NewJeans", "genre": "K-pop", "tags": ["nostalgic", "trending"]},
                {"name": "How You Like That", "artist": "BLACKPINK", "genre": "K-pop", "tags": ["energetic", "party"]},
                {"name": "Spring Day", "artist": "BTS", "genre": "K-pop", "tags": ["emotional", "classic"]},
                {"name": "Through the Night", "artist": "IU", "genre": "Ballad", "tags": ["calm", "emotional"]},
            ],
        }
