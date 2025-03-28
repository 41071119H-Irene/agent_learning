import os
from dotenv import load_dotenv
import asyncio

# 載入 .env 檔案中的環境變數
load_dotenv()

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer

async def run_chat():
    # 從 .env 讀取 Gemini API 金鑰
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    
    # 使用 Gemini API，指定 model 為 "gemini-1.5-flash-8b"
    model_client = OpenAIChatCompletionClient(
        model="gemini-1.5-flash-8b",
        api_key=gemini_api_key,
    )
    
    # 建立各代理人
    assistant = AssistantAgent("assistant", model_client)  # 原本的 AI 助理
    user_bot = AssistantAgent("user_bot", model_client)  # 取代 UserProxyAgent 的 AI 代理人
    web_surfer = MultimodalWebSurfer("web_surfer", model_client)  # 負責網頁搜尋
    
    # 當對話中出現 "exit" 時即終止對話
    termination_condition = MaxMessageTermination(max_messages=10)
    
    # 建立一個循環團隊，讓各代理人依序參與討論
    team = RoundRobinGroupChat(
        [web_surfer, assistant, user_bot],  # ✅ 改成 user_bot，而非 user_proxy
        termination_condition=termination_condition
    )
    
    # 啟動團隊對話，任務是「搜尋 Gemini 的相關資訊，並撰寫一份簡短摘要」
    await Console(team.run_stream(task="請搜尋 Gemini 的相關資訊，並撰寫一份簡短摘要。"))

# 避免 Windows 上的 asyncio 錯誤
if __name__ == '__main__':
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_chat())
    finally:
        loop.close()  # ✅ 確保資源釋放，避免 `I/O operation on closed pipe`
