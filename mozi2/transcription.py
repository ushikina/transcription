import streamlit as st
import whisper
import tempfile
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai


# Gemini APIキーを設定
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Whisperモデルのキャッシュ
@st.cache_resource
def get_whisper_model():
    return whisper.load_model("base")


# キャッシュクリアボタン
st.sidebar.button('Cache clear', on_click=st.cache_resource.clear)
st.sidebar.divider()


main_col1, main_col2 = st.columns([5, 1])
with main_col1:
# mainタイトル
    st.write("Whisper & GeminiAI ver.")
    st.title("🎙️ 文字起こし")

with main_col2:
# 画面クリアボタン
    if st.button("Data clear"):
        st.session_state["clear_triggered"] = True
# セッション状態のクリア/強制的リロード
if st.session_state.get("clear_triggered"):
    st.session_state.clear()
    st.rerun()


# 📁 音声ファイルアップロード
uploaded_file = st.sidebar.file_uploader("音声ファイルをアップロード", type=["mp3", "wav", "m4a"])

# 開始ボタン
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        start_clicked = st.button("🚀START", use_container_width=True)


# Whisper + Gemini 同時実行
if uploaded_file and start_clicked:
    st.session_state["uploaded_file"] = uploaded_file 
    start_time = time.time()  # 開始時間を記録
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Whisper進捗バー
    whisper_progress = st.progress(0, text="Whisperを初期化中...")
    model = get_whisper_model()
    whisper_progress.progress(25, "Whisperモデル読み込み完了")
    time.sleep(0.5)
    whisper_progress.progress(50, "音声を処理中...")

    result = model.transcribe(tmp_path, language="ja")
    transcript = result["text"]
    time.sleep(0.5)
    whisper_progress.progress(100, "Whisper完了 ✅")

    st.session_state["transcript"] = transcript
    st.session_state["show_whisper"] = True

    # Gemini進捗バー
    gemini_progress = st.progress(0, text="Geminiを初期化中...")
    gmodel = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
    gemini_progress.progress(30, "プロンプト作成中...")

    prompt = f"""
以下の文字起こしは医療現場での会話です。
ひらがなやカタカナ表記が含まれていますが、適切な漢字や専門用語（医学用語）を使用し、
わかる範囲で構わないので出来るだけ原文の意味を変えずに、自然で読みやすい日本語の文章体に変換してください。
箇条書きや箇条書き風の表現は避け、段落を使って文章を構成してください。
話し言葉の特徴（例：「ええと」「あの」「えー」など）は適宜修正し、よりフォーマルな表現に置き換えてください。
ただし、話し言葉によって強調されているニュアンスは保持するようにしてください。
専門用語はそのまま使用してください。
また、文脈が明確になるように、適宜接続詞や指示語を追加してください。
変換後のテキストは、元のトランスクリプトの内容をすべて網羅している必要があります。
一度に出力できない場合は、複数回に分割して出力してください。
入力した一連の文の文章体への変換が全て終了した場合は、「終了」と出力してください。

「{transcript}」
"""
    time.sleep(0.5)
    gemini_progress.progress(50, "Geminiが応答を生成中...")
    response = gmodel.generate_content(prompt)
    corrected = response.text
    time.sleep(0.5)
    gemini_progress.progress(100, "Gemini完了 ✅")

    st.session_state["corrected"] = corrected
    st.session_state["show_gemini"] = True

    # 経過時間を測定
    elapsed_time = time.time() - start_time  
    st.session_state["elapsed_time"] = elapsed_time


# オーディオ表示
if "uploaded_file" in st.session_state:
    st.divider()
    st.audio(st.session_state["uploaded_file"])

# Whisper結果表示
if st.session_state.get("show_whisper", False):
    st.divider()
    whisper_col1, whisper_col2 = st.columns([8, 1])
    with whisper_col1:
        st.subheader("🎧 Whisper")
    with whisper_col2:
        st.download_button(
            label="DL.txt",
            data=st.session_state["transcript"],
            file_name="whisper_transcript.txt",
            mime="text/plain",
            key="whisper_dl"
        )
    st.text_area(
        label="Whisperの処理結果", 
        value=st.session_state["transcript"], 
        height=100, 
        label_visibility="collapsed")

# Gemini結果表示
if st.session_state.get("show_gemini", False):
    st.divider()
    gemini_col1, gemini_col2 = st.columns([8, 1])
    with gemini_col1:
        st.subheader("🩺 Gemini/補正")
    with gemini_col2:
        st.download_button(
            label="DL.txt",
            data=st.session_state["corrected"],
            file_name="gemini_transcript.txt",
            mime="text/plain",
            key="gemini_dl"
        )
    st.text_area(
        label="Geminiの処理結果", 
        value=st.session_state["corrected"], 
        height=100,
        label_visibility="collapsed")

# 処理にかかった時間を表示
if "elapsed_time" in st.session_state:
    st.sidebar.divider()
    total_sec = int(st.session_state["elapsed_time"])
    minutes, seconds = divmod(total_sec, 60)
    st.sidebar.markdown(f"⏱️ 所要時間: `{minutes}分{seconds}秒`")

