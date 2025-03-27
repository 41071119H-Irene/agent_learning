import os
import asyncio
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from snownlp import SnowNLP
from dotenv import load_dotenv
import io

matplotlib.use('Agg')  # 使用非 GUI 的後端
matplotlib.rc('font', family='Microsoft JhengHei')

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.messages import TextMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()
semaphore = asyncio.Semaphore(1)  # 限制最大併發數

def generate_mood_trend_plot(user_id, user_entries):
    """
    生成用戶的心情趨勢圖，包含 SnowNLP 心情小語分析，並存入 `moodtrend/` 資料夾。
    """
    output_dir = "static/moodtrend"
    os.makedirs(output_dir, exist_ok=True)

    user_entries["日期"] = pd.to_datetime(user_entries["日期"])
    user_entries = user_entries.sort_values("日期")
    user_entries["心情指數"] = pd.to_numeric(user_entries["心情指數"], errors="coerce")
    user_entries["心情小語分析"] = user_entries["心情小語"].apply(lambda text: SnowNLP(text).sentiments * 9 + 1)
    avg_mood = user_entries["心情指數"].mean()

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=user_entries["日期"], y=user_entries["心情指數"], marker="o", label="心情指數", color="blue", ci=None)
    sns.lineplot(x=user_entries["日期"], y=user_entries["心情小語分析"], marker="o", label="心情小語分析", color="red", ci=None)
    plt.axhline(y=avg_mood, color='orange', linestyle='--', label=f"平均心情指數 ({avg_mood:.2f})")
    plt.xlabel("日期")
    plt.ylabel("心情指數")
    plt.title(f"用戶 {user_id} 的心情趨勢")
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.ylim(1, 10)

    output_path = os.path.join(output_dir, f"mood_trend_{user_id}.png")
    plt.savefig(output_path)
    plt.close()

    return output_path

async def _process_user_diary(user_id, user_entries, model_client, termination_condition):
    """處理單一用戶的日記，生成心情趨勢圖，並與 AI 進行互動分析。"""
    trend_image = generate_mood_trend_plot(user_id, user_entries)
    
    prompt = (
        f"目前正在處理用戶 {user_id} 的日記內容，共 {len(user_entries)} 則。\n"
        f"以下為該用戶的日記內容:\n{user_entries.to_dict(orient='records')}\n\n"
        "請根據以上內容進行分析，並提供該用戶專屬的正向思考建議，包含：\n"
        "  1. 分析該用戶的情緒與思考模式；\n"
        "  2. 提供實際可行的行動方案來改善負面情緒；\n"
        "  3. 訓練 AI 教練，使其能夠與該用戶進行個性化互動。"
    )

    analysis_agent = AssistantAgent("analysis_agent", model_client)
    coaching_agent = AssistantAgent("coaching_agent", model_client)
    
    team = RoundRobinGroupChat(
        [analysis_agent, coaching_agent],  # 移除 UserProxyAgent，僅保留 AI 代理
        termination_condition=termination_condition
    )

    messages = []
    async for event in team.run_stream(task=prompt):
        if isinstance(event, TextMessage):
            print(f"[{event.source}] => {event.content}\n")
            messages.append({
                "user_id": user_id,
                "source": event.source,
                "content": event.content,
                "type": event.type,
                "trend_image": trend_image
            })

    return messages

async def process_user_diary(user_id, user_entries, model_client, termination_condition):
    async with semaphore:
        return await _process_user_diary(user_id, user_entries, model_client, termination_condition)

async def main():
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("請檢查 .env 檔案中的 GEMINI_API_KEY。")
        return

    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",
        api_key=gemini_api_key,
    )
    
    termination_condition = TextMentionTermination("terminate")
    
    csv_file_path = "user_diary.csv"
    df = pd.read_csv(csv_file_path)
    
    df.rename(columns={"用戶ID": "user_id"}, inplace=True)

    if "user_id" not in df.columns:
        print("CSV 檔案缺少 `user_id` 欄位，請確認數據格式。")
        return
    
    user_groups = df.groupby("user_id")
    
    tasks = [
        asyncio.create_task(process_user_diary(user_id, user_entries, model_client, termination_condition))
        for user_id, user_entries in user_groups
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    all_messages = [msg for batch in results if isinstance(batch, list) for msg in batch]

    df_log = pd.DataFrame(all_messages)
    output_file = "personalized_positive_thinking_log.csv"
    df_log.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"已將個人化建議輸出為 {output_file}")

if __name__ == '__main__':
    asyncio.run(main())