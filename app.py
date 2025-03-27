import os
import asyncio
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from dotenv import load_dotenv
from EMOwithSnow import generate_mood_trend_plot, process_user_diary
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 載入環境變數
load_dotenv()

# Flask 應用程式
app = Flask(__name__)

# 設定靜態檔案夾
UPLOAD_FOLDER = "static/moodtrend"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
CSV_FILE_PATH = "user_diary.csv"

# 初始化 AI 客戶端
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("請設定 GEMINI_API_KEY 環境變數")

model_client = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    api_key=gemini_api_key
)

# 🟢 分析用戶日記
async def analyze_user_diary(user_id):
    if not os.path.exists(CSV_FILE_PATH):
        return None, None, None

    df = pd.read_csv(CSV_FILE_PATH)
    df.rename(columns={"用戶ID": "user_id"}, inplace=True)

    if user_id not in df["user_id"].unique():
        return None, None, None

    user_entries = df[df["user_id"] == user_id]
    trend_image_path = generate_mood_trend_plot(user_id, user_entries)

    # 設定終止條件
    termination_condition = None
    messages = await process_user_diary(user_id, user_entries, model_client, termination_condition)

    analysis_results = [msg["content"] for msg in messages[:3]] if messages else ["無數據"] * 3
    recommendations = ["練習正向思考", "多參與社交活動", "建立健康生活習慣"]

    return analysis_results, recommendations, trend_image_path

# 🟢 首頁
@app.route('/')
def index():
    return render_template('index.html')

# 🟢 提供靜態圖片
@app.route('/static/moodtrend/<filename>')
def mood_trend_image(filename):
    return send_from_directory("static/moodtrend", filename)

# 🟢 上傳新日記並分析
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files.get('file')

    if not file:
        return jsonify({'error': '沒有上傳文件'}), 400

    # 儲存 CSV
    file.save(CSV_FILE_PATH)

    # 取得 user_id，假設固定為 1
    user_id = 1

    # 執行分析
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    analysis_results, recommendations, trend_image_path = loop.run_until_complete(analyze_user_diary(user_id))

    if not analysis_results:
        return jsonify({'error': '分析失敗'}), 500

    # ✅ 修正圖片 URL
    mood_trend_image_url = f"/moodtrend/mood_trend_{user_id}.png"

    return jsonify({
        'analysis': analysis_results,
        'recommendations': recommendations,
        'mood_trend_image': mood_trend_image_url
    })


if __name__ == "__main__":
    app.run(debug=True)
