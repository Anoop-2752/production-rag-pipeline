import os
from dotenv import load_dotenv

load_dotenv()

# ── Test Groq ──────────────────────────────────────────
def test_groq():
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage

    try:
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile"
        )
        response = llm.invoke([HumanMessage(content="Say 'Groq is working!' and nothing else.")])
        print("✅ Groq:", response.content)
    except Exception as e:
        print("❌ Groq failed:", str(e))


# ── Test LangSmith ─────────────────────────────────────
def test_langsmith():
    import requests

    api_key = os.getenv("LANGCHAIN_API_KEY")
    headers = {"x-api-key": api_key}

    try:
        response = requests.get(
            "https://api.smith.langchain.com/api/v1/workspaces?limit=1",
            headers=headers
        )
        if response.status_code == 200:
            print("✅ LangSmith: Connected successfully")
        else:
            print(f"❌ LangSmith failed: Status {response.status_code} — {response.text}")
    except Exception as e:
        print("❌ LangSmith failed:", str(e))


if __name__ == "__main__":
    print("\n🔍 Testing API Keys...\n")
    test_groq()
    test_langsmith()
    print("\nDone!\n")