"""MoodTune AI Streamlit application.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from database import MoodDatabase
from image_analyzer import ImageAnalyzer
from mood_engine import SmartMoodEngine
from spotify_recommender import SpotifyRecommender
from utils import (
    add_bg_glow,
    build_playlist_pdf,
    css,
    get_image_hash,
    image_to_base64,
    normalize_preferred_genre,
)


st.set_page_config(
    page_title="MoodTune AI",
    page_icon="MT",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(css(), unsafe_allow_html=True)
add_bg_glow()


@st.cache_resource(show_spinner=False)
def get_analyzer() -> ImageAnalyzer:
    return ImageAnalyzer()


@st.cache_resource(show_spinner=False)
def get_engine() -> SmartMoodEngine:
    return SmartMoodEngine()


@st.cache_resource(show_spinner=False)
def get_recommender() -> SpotifyRecommender:
    return SpotifyRecommender()


@st.cache_resource(show_spinner=False)
def get_database() -> MoodDatabase:
    return MoodDatabase()


def show_image(image: Any, **kwargs: Any) -> None:
    """Render images across Streamlit versions.

    Older Streamlit releases use ``use_column_width`` while newer releases use
    ``use_container_width``.
    """

    try:
        st.image(image, use_container_width=True, **kwargs)
    except TypeError:
        st.image(image, use_column_width=True, **kwargs)


def render_metric_cards(analysis: dict[str, Any]) -> None:
    cols = st.columns(4)
    cards = [
        ("Main emotion", analysis.get("main_emotion", "Unknown")),
        ("Recommended genre", analysis.get("recommended_genre", "Mixed")),
        ("Mood score", f"{analysis.get('mood_score', 0)}%"),
        ("Playlist title", analysis.get("playlist_title", "Untitled vibe")),
    ]
    for col, (label, value) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <span>{label}</span>
                    <strong>{value}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_feature_panel(features: dict[str, Any]) -> None:
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("Extracted Features")
        st.markdown(
            f"""
            <div class="feature-box">
                <div><b>Scene understanding</b><p>{features.get("scene_understanding", "No scene summary available.")}</p></div>
                <div><b>Emotional interpretation</b><p>{features.get("emotional_interpretation", "No interpretation available.")}</p></div>
                <div><b>Aesthetic vibe</b><p>{features.get("aesthetic_style", "Natural")}</p></div>
                <div><b>Social context</b><p>{features.get("social_context", "Unknown")}</p></div>
                <div><b>Activity recognition</b><p>{", ".join(features.get("activities", []) or ["Unclear"])}</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Contextual Tags")
        tags = features.get("contextual_tags", [])
        if tags:
            st.markdown(
                "<div class='tag-wrap'>"
                + "".join(f"<span class='tag'>{tag}</span>" for tag in tags)
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.info("No tags generated yet.")

        color_items = features.get("color_psychology", [])
        if color_items:
            st.markdown("#### Color Psychology")
            for item in color_items:
                st.markdown(
                    f"<div class='color-row'><span style='background:{item['hex']}'></span>"
                    f"<b>{item['name']}</b><small>{item['emotion']}</small></div>",
                    unsafe_allow_html=True,
                )


def render_mood_analysis(analysis: dict[str, Any], features: dict[str, Any]) -> None:
    st.subheader("Mood Analysis")
    render_metric_cards(analysis)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### Mood Tags")
        st.markdown(
            "<div class='tag-wrap large-tags'>"
            + "".join(f"<span class='tag'>{tag}</span>" for tag in analysis.get("mood_tags", []))
            + "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("#### Vibe Description")
        st.write(analysis.get("vibe_description", "No vibe description available."))

    with col_b:
        st.markdown("#### Smart Mood Engine")
        score_rows = [
            {"Signal": name.title(), "Score": round(score, 1)}
            for name, score in analysis.get("score_breakdown", {}).items()
        ]
        if score_rows:
            for row in score_rows:
                st.progress(
                    min(int(row["Score"]), 100),
                    text=f"{row['Signal']}: {row['Score']}%",
                )
        else:
            st.info("Mood scoring data was not generated.")

    with st.expander("AI Generated Caption", expanded=True):
        st.write(features.get("caption", "Caption not available."))


def render_song_card(song: dict[str, Any], idx: int) -> None:
    image = song.get("album_cover") or "https://placehold.co/220x220/111827/f8fafc?text=MoodTune"
    artist = song.get("artist", "Unknown artist")
    name = song.get("name", "Unknown song")
    match = song.get("genre_match", 0)
    reason = song.get("why", "This song matches the detected mood and listening preferences.")
    url = song.get("spotify_url") or song.get("search_url") or "https://open.spotify.com/search"

    with st.container(border=True):
        col_img, col_body, col_action = st.columns([0.9, 2.2, 1.1])
        with col_img:
            show_image(image)
        with col_body:
            st.markdown(f"#### {idx}. {name}")
            st.caption(artist)
            st.progress(min(int(match), 100), text=f"{match}% genre match")
            st.write(reason)
            preview = song.get("preview_url")
            if preview:
                st.audio(preview)
        with col_action:
            st.link_button("Open in Spotify", url, use_container_width=True)
            uri = song.get("spotify_uri")
            if uri:
                st.code(uri, language=None)


def render_recommendations(
    songs: list[dict[str, Any]],
    playlists: list[dict[str, Any]],
    recommender: SpotifyRecommender,
    analysis: dict[str, Any],
) -> None:
    st.subheader("Recommended Songs")
    if not songs:
        st.warning("No songs were found for this mood. Try a broader genre or a different language.")
        return

    top_cols = st.columns(5)
    for i, song in enumerate(songs[:5]):
        with top_cols[i]:
            show_image(song.get("album_cover") or "https://placehold.co/300x300/111827/f8fafc?text=Song")
            st.markdown(f"**{song.get('name', 'Song')}**")
            st.caption(song.get("artist", "Artist"))

    st.markdown("#### Full Recommendations")
    for idx, song in enumerate(songs, start=1):
        render_song_card(song, idx)

    st.subheader("Spotify Player")
    first_track_id = songs[0].get("spotify_id")
    if first_track_id:
        components.iframe(
            f"https://open.spotify.com/embed/track/{first_track_id}?utm_source=generator",
            height=160,
        )
    else:
        st.info("Spotify embed is available when API credentials return a track id.")

    st.subheader("Similar Mood Playlist")
    playlist_cols = st.columns(3)
    for i, playlist in enumerate(playlists[:3]):
        with playlist_cols[i]:
            show_image(
                playlist.get("image") or "https://placehold.co/420x240/111827/f8fafc?text=Playlist",
            )
            st.markdown(f"**{playlist.get('name', 'Mood playlist')}**")
            st.caption(playlist.get("owner", "Spotify"))
            st.link_button("Open playlist", playlist.get("url", "https://open.spotify.com/search"), use_container_width=True)

    pdf_bytes = build_playlist_pdf(analysis, songs)
    st.download_button(
        "Download playlist as PDF",
        data=pdf_bytes,
        file_name=f"{analysis.get('playlist_title', 'moodtune-playlist').lower().replace(' ', '-')}.pdf",
        mime="application/pdf",
    )

    if recommender.can_export_playlist:
        with st.expander("Export playlist to Spotify account"):
            public = st.toggle("Make playlist public", value=False)
            if st.button("Create Spotify playlist"):
                with st.spinner("Creating playlist in Spotify..."):
                    result = recommender.create_playlist(
                        title=analysis.get("playlist_title", "MoodTune AI Playlist"),
                        description=analysis.get("vibe_description", "Generated by MoodTune AI."),
                        track_uris=[song["spotify_uri"] for song in songs if song.get("spotify_uri")],
                        public=public,
                    )
                if result.get("ok"):
                    st.success("Playlist created.")
                    st.link_button("Open exported playlist", result["url"])
                else:
                    st.error(result.get("error", "Spotify playlist export failed."))


def render_history(db: MoodDatabase) -> None:
    st.subheader("History of Previous Searches")
    rows = db.get_history(limit=12)
    if not rows:
        st.info("Your image mood history will appear here after the first analysis.")
        return

    for row in rows:
        with st.container(border=True):
            cols = st.columns([0.75, 2.2, 1])
            with cols[0]:
                if row.get("thumbnail_b64"):
                    show_image(f"data:image/jpeg;base64,{row['thumbnail_b64']}")
            with cols[1]:
                st.markdown(f"**{row.get('main_emotion', 'Unknown').title()}**")
                st.caption(row.get("created_at", ""))
                st.write(row.get("vibe_description", ""))
                st.markdown(
                    "<div class='tag-wrap'>"
                    + "".join(f"<span class='tag'>{tag}</span>" for tag in row.get("mood_tags", [])[:6])
                    + "</div>",
                    unsafe_allow_html=True,
                )
            with cols[2]:
                recs = row.get("recommendations", [])
                if recs:
                    st.caption("Top song")
                    st.write(recs[0].get("name", "Song"))
                    st.link_button("Spotify", recs[0].get("spotify_url", "https://open.spotify.com/search"))


def analyze_and_recommend(
    image: Image.Image,
    image_name: str,
    image_hash: str,
    language: str,
    genre_pref: str,
    style: str,
    mood_refinement: str,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    analyzer = get_analyzer()
    engine = get_engine()
    recommender = get_recommender()
    db = get_database()

    progress = st.progress(0, text="Reading image signals...")
    time.sleep(0.1)
    features = analyzer.analyze(image)
    progress.progress(30, text="Understanding mood, social context, and colors...")
    analysis = engine.generate_mood_profile(
        features=features,
        preferred_language=language,
        preferred_genre=normalize_preferred_genre(genre_pref),
        recommendation_style=style,
        mood_refinement=mood_refinement,
    )
    progress.progress(55, text="Searching Spotify for matching songs...")
    songs = recommender.recommend_songs(
        mood_profile=analysis,
        preferred_language=language,
        preferred_genre=normalize_preferred_genre(genre_pref),
        recommendation_style=style,
        limit=8,
    )
    progress.progress(75, text="Building similar playlists...")
    playlists = recommender.search_playlists(analysis, preferred_language=language, limit=6)

    embedding = engine.build_visual_mood_embedding(features, analysis)
    similar = db.get_similar_searches(embedding, limit=3)
    db.save_search(
        image_hash=image_hash,
        image_name=image_name,
        thumbnail_b64=image_to_base64(image, max_size=(360, 360)),
        features=features,
        analysis=analysis,
        recommendations=songs,
        language=language,
        genre_pref=genre_pref,
        style=style,
        embedding=embedding,
    )
    progress.progress(100, text="MoodTune AI is ready.")
    time.sleep(0.2)
    progress.empty()
    return features, analysis, songs, playlists, similar


def main() -> None:
    st.markdown(
        """
        <section class="hero">
            <div>
                <span class="eyebrow">AI image mood to music engine</span>
                <h1>MoodTune AI</h1>
                <p>Upload a moment. Get the soundtrack that fits its emotion, color, scene, and social energy.</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Controls")
        language = st.selectbox(
            "Preferred language",
            ["English", "Hindi", "Malayalam", "Tamil", "Korean"],
            index=0,
        )
        genre_pref = st.selectbox(
            "Genre preference",
            ["Auto", "Pop", "Hip-Hop", "Rock", "Electronic", "Dance", "Indie", "Lo-fi", "R&B", "Bollywood", "K-pop", "Classical"],
        )
        style = st.radio(
            "Recommendation style",
            ["Trending", "Classic", "Indie", "Lo-fi", "Viral"],
            horizontal=False,
        )
        mood_refinement = st.text_input("Mood refinement", placeholder="Add a word like roadtrip, heartbreak, gym...")

        st.divider()
        spotify = get_recommender()
        if spotify.is_configured:
            st.success("Spotify API connected")
        else:
            st.warning("Spotify API credentials are not configured. The app will use smart Spotify search links.")

    upload_tab, webcam_tab = st.tabs(["Upload Image", "Webcam Mood"])

    uploaded_file = None
    camera_image = None
    with upload_tab:
        uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg", "webp"])
    with webcam_tab:
        camera_image = st.camera_input("Real-time webcam mood detection")

    source = uploaded_file or camera_image
    if not source:
        st.info("Upload or capture an image to start mood analysis.")
        render_history(get_database())
        return

    image = Image.open(source).convert("RGB")
    image_name = getattr(source, "name", "webcam-capture.jpg")
    image_hash = get_image_hash(image)
    analysis_key = f"{image_hash}:{language}:{genre_pref}:{style}:{mood_refinement}"

    col_preview, col_result = st.columns([0.9, 1.4], gap="large")
    with col_preview:
        st.subheader("Image Preview")
        show_image(image)

    cached_result = st.session_state.get("last_result")
    if st.session_state.get("last_analysis_key") == analysis_key and cached_result:
        features, analysis, songs, playlists, similar = cached_result
        with col_result:
            st.success("Using the latest MoodTune AI analysis for this image.")
    else:
        with col_result:
            with st.status("Processing image with MoodTune AI...", expanded=True) as status:
                features, analysis, songs, playlists, similar = analyze_and_recommend(
                    image=image,
                    image_name=image_name,
                    image_hash=image_hash,
                    language=language,
                    genre_pref=genre_pref,
                    style=style,
                    mood_refinement=mood_refinement,
                )
                status.update(label="Analysis complete", state="complete", expanded=False)
        st.session_state["last_analysis_key"] = analysis_key
        st.session_state["last_result"] = (features, analysis, songs, playlists, similar)

    render_feature_panel(features)
    render_mood_analysis(analysis, features)
    render_recommendations(songs, playlists, get_recommender(), analysis)

    if similar:
        st.subheader("Similar Images Recommendation")
        st.caption("Past uploads with nearby visual mood embeddings.")
        cols = st.columns(min(3, len(similar)))
        for i, item in enumerate(similar):
            with cols[i]:
                if item.get("thumbnail_b64"):
                    show_image(f"data:image/jpeg;base64,{item['thumbnail_b64']}")
                st.markdown(f"**{item.get('main_emotion', 'Mood').title()}**")
                st.caption(f"{round(item.get('similarity', 0) * 100)}% similar")
                recs = item.get("recommendations", [])
                if recs:
                    st.write(recs[0].get("name", "Recommended song"))

    render_history(get_database())


if __name__ == "__main__":
    main()
