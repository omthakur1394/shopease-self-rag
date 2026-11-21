import os
import re
import json
import hashlib
import asyncio
import numpy as np
import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "")
CACHE_ENABLED = os.getenv("REC_CACHE_ENABLED", "true").lower() == "true"
SIM_THRESHOLD = float(os.getenv("REC_CACHE_THRESHOLD", "0.93"))
TTL_SECONDS = int(os.getenv("REC_CACHE_TTL", "604800"))
MAX_ENTRIES = int(os.getenv("REC_CACHE_MAX", "300"))
PREFIX = "rec"
INDEX_KEY = "rec:index"
SKIP_WORDS = ["place order", "place the order", "confirm", "cancel", "return", "refund", "track", "ord-"]

stats = {"hits": 0, "misses": 0}


def cache_stats():
    total = stats["hits"] + stats["misses"]
    rate = round(stats["hits"] / total, 3) if total else 0.0
    return {"hits": stats["hits"], "misses": stats["misses"], "hit_rate": rate}


class RecommendationCache:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.client = None
        if CACHE_ENABLED and REDIS_URL:
            self.client = redis.from_url(REDIS_URL, decode_responses=True)
        print("[cache] enabled" if self.client else "[cache] disabled")

    def normalize(self, text):
        cleaned = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
        return " ".join(cleaned.split())

    def key_for(self, norm):
        digest = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:24]
        return f"{PREFIX}:{digest}"

    def numbers_of(self, norm):
        return sorted(re.findall(r"\d+", norm))

    def should_skip(self, norm):
        if not norm:
            return True
        for word in SKIP_WORDS:
            if word in norm:
                return True
        return False

    async def embed(self, text):
        vec = await asyncio.to_thread(self.embeddings.embed_query, text)
        arr = np.array(vec, dtype=np.float32)
        length = np.linalg.norm(arr)
        if length == 0:
            return arr
        return arr / length

    async def get(self, query):
        if not self.client:
            return None
        norm = self.normalize(query)
        if self.should_skip(norm):
            return None
        try:
            raw = await self.client.get(self.key_for(norm))
            if raw:
                stats["hits"] += 1
                print("[cache] HIT exact")
                return json.loads(raw)["answer"]

            members = await self.client.smembers(INDEX_KEY)
            keys = list(members)
            if keys:
                values = await self.client.mget(keys)
                q_vec = await self.embed(norm)
                q_nums = self.numbers_of(norm)
                best_score = 0.0
                best_answer = None
                for k, v in zip(keys, values):
                    if v is None:
                        await self.client.srem(INDEX_KEY, k)
                        continue
                    entry = json.loads(v)
                    if entry.get("nums", []) != q_nums:
                        continue
                    score = float(np.dot(q_vec, np.array(entry["emb"], dtype=np.float32)))
                    if score > best_score:
                        best_score = score
                        best_answer = entry["answer"]
                if best_answer is not None and best_score >= SIM_THRESHOLD:
                    stats["hits"] += 1
                    print(f"[cache] HIT semantic {best_score:.3f}")
                    return best_answer
        except Exception as e:
            print("[cache get error]", type(e).__name__)
        stats["misses"] += 1
        print("[cache] MISS")
        return None

    async def set(self, query, answer):
        if not self.client or not answer:
            return
        norm = self.normalize(query)
        if self.should_skip(norm):
            return
        try:
            count = await self.client.scard(INDEX_KEY)
            if count >= MAX_ENTRIES:
                return
            vec = await self.embed(norm)
            entry = {
                "answer": answer,
                "emb": vec.tolist(),
                "nums": self.numbers_of(norm),
            }
            key = self.key_for(norm)
            await self.client.set(key, json.dumps(entry), ex=TTL_SECONDS)
            await self.client.sadd(INDEX_KEY, key)
        except Exception as e:
            print("[cache set error]", type(e).__name__)