import os
import sys
import time
import requests
import numpy as np

import pandas as pd
from dotenv import load_dotenv

load_dotenv(".env", override=True)
load_dotenv(".env.local", override=True)

URL = os.environ.get("HF_API_URL", "https://omthakur1394-rag-api.hf.space/chat")
TOKEN_URL = os.environ.get("HF_TOKEN_URL", "https://omthakur1394-rag-api.hf.space/token")
HF_USERNAME = os.environ.get("HF_USERNAME", "webapp")
HF_PASSWORD = os.environ.get("HF_PASSWORD", "webapp")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"].strip()
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o-mini")
TAG = sys.argv[1] if len(sys.argv) > 1 else "before"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 25
print("[OPENAI] key prefix:", OPENAI_API_KEY[:7], "| judge:", JUDGE_MODEL)

ALL_QUESTIONS = [
    "How long do I have to return an electronics item?",
    "What happens if my phone is dead on arrival?",
    "Does ShopEase repair electronics in-house?",
    "What is the return window for clothing?",
    "Can I return innerwear or swimwear?",
    "What condition must fashion items be in to be returned?",
    "Can I return fresh vegetables that arrived spoiled?",
    "What is the minimum shelf life for packaged food at delivery?",
    "Can I return medicines after delivery?",
    "What minimum expiry period do medicines need at delivery?",
    "What do I do if my insulin arrived warm?",
    "What if I received the 9th edition of a book instead of the 10th?",
    "Within how many days must furniture damage be reported?",
    "What happens if a table leg is missing from my furniture order?",
    "When is TV or AC installation scheduled after delivery?",
    "How much wallet credit do I get if installation is delayed?",
    "What is Open Box Delivery?",
    "What should I do if some items are missing from my package?",
    "What happens if my order is lost in transit?",
    "What happens if I refuse a COD order at the door?",
    "Can I change my address after the order is dispatched?",
    "What do I get if I received a counterfeit product?",
    "How does the exchange policy work?",
    "What if the item was damaged during return pickup?",
    "My refund shows processed but I have not received the money.",
    "How long do refunds take for UPI and credit cards?",
    "I accidentally placed a duplicate order. What can I do?",
    "Does ShopEase refund the price difference if the price drops after purchase?",
    "Who is the Grievance Officer and how can I contact them?",
    "How do I file a formal complaint?",
    "Does RERA apply to my delayed delivery?",
    "How much compensation do I get for a delivery delayed by 5 days?",
    "How do I delete my account and data?",
    "Why was my account suspended and how can I appeal?",
    "Can I stack two coupon codes on one order?",
    "How long do I have to complete checkout on a flash sale item?",
    "When do EasePoints expire and what is their value?",
    "How does the referral program work?",
    "What are the benefits of ShopEase Plus?",
    "What is the delivery fee for orders below Rs. 499?",
    "How many delivery attempts are made before the package returns?",
    "What payment methods does ShopEase accept?",
    "What is the maximum order value for Cash on Delivery?",
    "How do I cancel an order?",
    "Can I modify my order after placing it?",
    "How can I contact customer support?",
    "How do I report a fraudulent seller?",
    "What does the Brand Authorized badge mean?",
    "What if the brand denies warranty because the seller is unauthorized?",
    "Who is responsible when a product does not match its listing?",
]

# pick LIMIT questions spread evenly across all topics (same ones every run)
QUESTIONS = []
if LIMIT >= len(ALL_QUESTIONS):
    QUESTIONS = ALL_QUESTIONS
else:
    step = len(ALL_QUESTIONS) / LIMIT
    for n in range(LIMIT):
        QUESTIONS.append(ALL_QUESTIONS[int(n * step)])


def get_auth_token():
    try:
        r = requests.post(
            TOKEN_URL,
            data={"username": HF_USERNAME, "password": HF_PASSWORD, "grant_type": "password"},
            timeout=30,
        )
        if r.ok:
            return r.json().get("access_token")
        print(f"[WARN] token failed: {r.status_code} {r.text}")
    except Exception as e:
        print(f"[WARN] token error: {e}")
    return None


auth_token = get_auth_token()
print("[AUTH] token acquired" if auth_token else "[WARN] no token, calls may get 401")


def call(q, idx):
    global auth_token
    for attempt in range(2):
        t = time.perf_counter()
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        payload = {
            "chat": q,
            "thread_id": f"eval_{TAG}_{idx}",
            "user_id": "eval_user",
            "username": "eval_user",
        }
        try:
            r = requests.post(URL, headers=headers, json=payload, timeout=120)
            ms = (time.perf_counter() - t) * 1000
            if r.status_code == 401 and attempt == 0:
                auth_token = get_auth_token()
                continue
            if r.status_code == 200:
                d = r.json()
                ans = d.get("res") or d.get("response") or d.get("answer") or ""
                raw_ctx = d.get("contexts") or d.get("retrieved_contexts") or []
                ctx = []
                for c in raw_ctx:
                    if isinstance(c, str):
                        ctx.append(c)
                    else:
                        ctx.append(str(c))
                return ans, ctx, ms, 200
            return "", [], ms, r.status_code
        except Exception:
            return "", [], (time.perf_counter() - t) * 1000, 0
    return "", [], 0, 401


print(f"\nBenchmark: {len(QUESTIONS)} questions | tag={TAG}")
rows = []
for i, q in enumerate(QUESTIONS, 1):
    a, c, ms, s = call(q, i)
    rows.append({"question": q, "answer": a, "contexts": c, "latency_ms": ms, "status": s})
    print(f"{i}/{len(QUESTIONS)} {ms:.0f} ms status={s} ctx={len(c)}")
    time.sleep(1)

df = pd.DataFrame(rows)
ok = df[(df.status == 200) & (df.answer != "")]

print("\n========== LATENCY ==========")
if len(ok):
    lat = ok.latency_ms.values
    print("avg ms:", round(lat.mean(), 1))
    print("p50 ms:", round(np.percentile(lat, 50), 1))
    print("p95 ms:", round(np.percentile(lat, 95), 1))
else:
    print("no successful queries")
print("success rate:", f"{len(ok)}/{len(df)}")

df.to_csv(f"latency_{TAG}.csv", index=False)
print(f"saved latency_{TAG}.csv")

rag = ok[ok.contexts.apply(len) > 0]
print("queries with contexts:", len(rag))

if len(rag):
    try:
        from ragas import evaluate, EvaluationDataset, RunConfig
        from ragas.metrics import faithfulness, AnswerRelevancy, LLMContextPrecisionWithoutReference
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI
        from langchain_huggingface import HuggingFaceEmbeddings

        samples = []
        for r in rag.itertuples():
            samples.append({
                "user_input": r.question,
                "response": r.answer,
                "retrieved_contexts": r.contexts,
            })
        ds = EvaluationDataset.from_list(samples)

        llm = LangchainLLMWrapper(
            ChatOpenAI(model=JUDGE_MODEL, api_key=OPENAI_API_KEY, temperature=0)
        )
        emb = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        )

        print(f"\n========== RAGAS ({JUDGE_MODEL}) ==========")
        res = evaluate(
            ds,
            metrics=[faithfulness, AnswerRelevancy(strictness=1), LLMContextPrecisionWithoutReference()],
            llm=llm,
            embeddings=emb,
            run_config=RunConfig(max_workers=4, timeout=180, max_retries=5),
            raise_exceptions=False,
        )
        sc = res.to_pandas()
        for col in sc.columns:
            if col not in ("user_input", "response", "retrieved_contexts"):
                print(col, round(sc[col].mean(), 3))
        sc.to_csv(f"ragas_{TAG}.csv", index=False)
        print(f"saved ragas_{TAG}.csv")
    except Exception as e:
        print(f"[RAGAS ERROR] {e}")
else:
    print("\nNo contexts returned by /chat. Add a 'contexts' field to the API response to run RAGAS.")