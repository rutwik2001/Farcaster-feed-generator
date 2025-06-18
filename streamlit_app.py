import streamlit as st
from pymongo import MongoClient
from operator import itemgetter
import time
import math
import os
import psycopg2
from psycopg2 import pool
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import time
from psycopg2.extras import DictCursor
import ast

model_name = "sentence-transformers/all-mpnet-base-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

db_pool = psycopg2.pool.SimpleConnectionPool(
    1, 20,
    st.secrets["PostgreSQL"]
    
)

# === CONFIGURATION ===
MONGO_URI = st.secrets["MONGO_URI"]
DB_NAME = "Social_Media"
USER_COLLECTION = "farcaster_users"
POST_COLLECTION = "farcaster_new_casts"
TOP_N_CREATORS = 200
POSTS_PER_CREATOR = 1
MIN_CREATOR_SCORE = 0.5


cryptoTopics = [
  "blockchain scalability solutions",
  "Ethereum gas optimization",
  "smart contract vulnerabilities",
  "layer 2 rollups explained",
  "proof of stake consensus mechanism",
  "zero-knowledge proofs in Web3",
  "decentralized identity systems",
  "Web3 storage protocols comparison",
  "interoperability between blockchains",
  "on-chain governance models",
  "how DAOs operate",
  "what is a decentralized application",
  "explainer on tokenomics models",
  "understanding staking vs slashing",
  "history of blockchain forks",
  "introduction to NFTs beyond art",
  "Web3 vs Web2 data ownership",
  "crypto regulation landscape in 2024",
  "ethical concerns in blockchain adoption",
  "Web3 and financial inclusion",
  "regulatory risks for DeFi platforms",
  "privacy implications of public blockchains",
  "compliance challenges in crypto KYC",
  "limitations of current DeFi platforms",
  "scalability challenges in Web3",
  "risks of algorithmic stablecoins",
  "centralization risks in popular blockchains",
  "energy consumption of proof of work systems",
  "critique of token-based governance",
  "blockchain use cases in supply chains",
  "academic research on NFT markets",
  "long-term viability of Web3 protocols",
  "network effects in decentralized platforms",
  "adoption barriers in crypto ecosystems"
]

womenTopics = [
    "Women In STEM", "Women in Medicine", "Women in Tech", "Women in Business",
    "Women History Month 2025", "Women History Month", "Women Activist", "Women in Science",
    "Women's History", "Women in Leadership", "Women in Art", "Gender Equality",
    "Feminism", "Empowered Women", "Women Supporting Women", "IWD 2025", "Empower Women",
    "Women's Rights", "Women Entrepreneurs", "Successful Women", "Women Led", "Women Lawyers",
    "Women Empowerment", "Women Power", "Female Entrepreneur", "Women Owned", "Women In Law",
    "Women Owned Business", "Women Leaders", "Inspiring Women", "Women Leading Change",
    "Women Who Code", "Girls in STEM", "Women's Art", "Feminist", "Women in Marketing",
    "Women in sports", "women's sport"
]

w30Topics = [
    "Trailer", "Instagram", "Netflix", "Pet Memes", "amazon prime", "Disney+", "Twitter Memes",
    "Must Watch Movies", "K-pop", "K-drama", "Mortal", "S8UL", "Valorant", "Fortnite",
    "Counter-Strike", "PUBG", "League of Legends", "Apex Legends", "Call of Duty", "VCT", "BGMI",
    "FaZe Clan", "Shroud", "T1 Esports", "Team Soul", "Hydra Esports", "Ninja", "G2 Esports",
    "Pokimane", "ScoutOP", "Among Us", "Meal Prep", "Aesthetic Food", "Fitness", "Workouts",
    "FitTok", "Morning Routine", "DIY", "Life Hacks", "2000s Nostalgia", "Travel in India",
    "Student Budgeting Tips", "Book Recommendations", "Booktok", "Celebrity Fashion Breakdowns",
    "Red Carpet Looks", "Indie Music Discoveries", "Mental Health Awareness", "Social Media Detox",
    "Mindfulness", "iron man", "spiderman", "Harry Potter", "Vintage Clothing Trends", "Formula 1",
    "Football", "Cricket", "Tech Advancements", "Smartwatch Reviews", "Smartphone Reviews",
    "Entrepreneur Stories", "AI", "Software Engineer", "Leetcode", "Hackerrank", "Vibe coding",
    "Open Source Tools", "VS Code Extensions", "GitHub Repos", "Chrome Extensions",
    "Stand-Up Comedy Shorts", "Minecraft", "Pinterest", "Internship Hacks", "Productivity",
    "udemy", "coursera", "linkedin", "Genshin", "meta", "google", "microsoft", "apple",
    "Architecture", "Nature", "landscapes", "hiking", "ghibli", "music", "themermaidscales",
    "Monster Hunter Wilds", "Assassin’s Creed", "DOOM: The Dark Ages", "Clair Obscur",
    "Kingdom Come: Deliverance II", "squid game", "Pokémon Legends", "Ghost of Yotei",
    "Shark Tank", "fashionsky"
]


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output['last_hidden_state']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask

def generate_embedding(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        model_output = model(**inputs, return_dict=True)
    embedding = mean_pooling(model_output, inputs["attention_mask"])
    embedding = F.normalize(embedding, p=2, dim=1)
    return embedding.squeeze().cpu().numpy()

def search_posts(topic_query, allowed_user_ids, user_scores_dict, similarity_threshold=0.3):
    conn = db_pool.getconn()
    try:
        cur = conn.cursor(cursor_factory=DictCursor)
        embedding = generate_embedding(topic_query)
        embedding_str = "[" + ",".join(map(str, embedding.tolist())) + "]"

        current_time_ms = int(time.time() * 1000)
        past_24h_ms = current_time_ms - (24 * 3600 * 1000)

        user_ids_tuple = tuple(allowed_user_ids)
        if len(user_ids_tuple) == 1:
            user_ids_tuple += ('',)

        sql = f"""
            SELECT s.post_id, s.username, s.handle, s.description, 1 - (s.vsearch <=> %s) AS similarity
            FROM public.social_search_2 s
            WHERE s.type = 'farcaster'
              AND s.username IN %s
              AND 1 - (s.vsearch <=> %s) >= %s
              AND s.time BETWEEN %s AND %s
        """
        cur.execute(sql, (embedding_str, user_ids_tuple, embedding_str, similarity_threshold, past_24h_ms, current_time_ms))
        rows = cur.fetchall()
        results = []
        for r in rows:
            user_id = r["handle"]
            if user_id in user_scores_dict:
                results.append(
                    dict(r) | {
                        "creator_score": user_scores_dict[user_id]
                    }
                )
        return results
    finally:
        db_pool.putconn(conn)

# === MongoDB Connection ===
@st.cache_resource
def get_mongo_client():
    return MongoClient(MONGO_URI)

client = get_mongo_client()
db = client[DB_NAME]
users_col = db[USER_COLLECTION]
posts_col = db[POST_COLLECTION]

# === Utility Functions ===
def log_norm_eng(score):
    try:
        return max(0.0, min(1.0, (math.log10(score) - math.log10(1e-9)) / (math.log10(1e-3) - math.log10(1e-9))))
    except:
        return 0.0

def squash(x):
    return 1 / (1 + math.exp(-10 * (x - 0.3)))

# === Main Logic ===
def generate_feed(organicThreshold, humane_scoreThreshold, engagement_scoreThreshold, postSimilarityScoreThreshold, weights, isCrypto, isWomen, isW30Feed, topicsInput, isCreator):
    now = int(time.time() * 1000)
    past_24h = now - (2 * 24 * 60 * 60 * 1000)

    recent_user_ids = posts_col.distinct("userId", {
        "timeInUnix": {"$gte": past_24h}
    })

    filtered_users = users_col.find({
        "userId": {"$in": recent_user_ids},
        "authenticityScore": {"$exists": True},
        "organicScore": {"$gte": organicThreshold},
        "humane_score": {"$gte": humane_scoreThreshold},
        "engagement_score": {"$gte": engagement_scoreThreshold},
        "postSimilarityScore": {"$lte": postSimilarityScoreThreshold}
    })
    
    user_scores = []
    user_dict = {}
    user_link = {}

    for user in filtered_users:
        norm_eng = squash(log_norm_eng(user.get("engagement_score", 1e-9)))
        score = round(
            weights["organic"] * user.get("organicScore", 0) +
            weights["humane"] * user.get("humane_score", 0) +
            weights["engagement"] * norm_eng +
            weights["similarity"] * (1 - user.get("postSimilarityScore", 1.0)), 6
        )
        user_scores.append({
        "userId": user["userId"],
        "creator_score": score,
        "handle": user["username"]
        })
        user_dict[user["username"]] = score
        user_link[user["userId"]] = f"https://firefly.social/profile/farcaster/{user["username"]}/feed"

    top_creators = sorted(user_scores, key=itemgetter("creator_score"), reverse=True)
    topics = ""
    if isCrypto:
        topics = ", ".join(cryptoTopics)
    if isWomen:
        topics = ", ".join(womenTopics)
    if isW30Feed:
        topics = ", ".join(w30Topics)
    if topicsInput:
        topics = topicsInput

    allowed_user_ids = [u["handle"] for u in top_creators]
    user_scores_dict = {u["handle"]: u["creator_score"] for u in top_creators}

    results = search_posts(topics, allowed_user_ids, user_scores_dict)
    if isCreator:
        results_sorted = sorted(results, key=lambda x: x["creator_score"], reverse=True)
    else:
        results_sorted = sorted(results, key=lambda x: x["similarity"], reverse=True)
    filtered_feed = [
        {
            "handle": post["handle"],
            "fireflyLink": f"https://firefly.social/profile/farcaster/{post['handle']}/feed",
            "hash": post["post_id"],
            "text": post.get("description", ""),
            "score": round(post.get("creator_score", 0), 4),
            "similarity": round(post.get('similarity', 0.0), 4)
        }
        for post in results_sorted if post.get("creator_score", 0) >= min_creator_score and post.get("description", "").strip()
    ]

    return filtered_feed

# === Streamlit Frontend ===
st.title("📡 Farcaster Feed Generator")

st.sidebar.header("🧠 Topic Selection")

sort_by = st.sidebar.radio(
    "Sort by:",
    ("Creator Score", "Similairty"),
    index=0
)

isCreator = False
if sort_by == "Creator Score":
    isCreator = True


topic_mode = st.sidebar.radio(
    "Select a topic mode (only one can be active):",
    ("Manual Input", "Crypto", "Women", "W30"),
    index=0
)

topicsInput = ""
isCrypto = isWomen = isW30Feed = False

if topic_mode == "Manual Input":
    topicsInput = st.sidebar.text_input("Enter topics (comma-separated)", value="")
elif topic_mode == "Crypto":
    isCrypto = True
elif topic_mode == "Women":
    isWomen = True
elif topic_mode == "W30":
    isW30Feed = True

st.sidebar.header("📈 Min Creator Score Threshold")
min_creator_score = st.sidebar.slider("Min Creator Score Threshold", 0.0, 1.0, 0.3, 0.01)

st.sidebar.header("📊 Set Score Thresholds")
organicThreshold = st.sidebar.slider("Organic Score Threshold", 0.0, 1.0, 0.1, 0.01)
humane_scoreThreshold = st.sidebar.slider("Humane Score Threshold", 0.0, 1.0, 0.1, 0.01)
engagement_scoreThreshold = st.sidebar.slider("Engagement Score Threshold", 0.0, 1.0, 0.0, 0.00001, format="%.5f")
postSimilarityScoreThreshold = st.sidebar.slider("Max Post Similarity", 0.0, 1.0, 0.9, 0.01)

st.sidebar.header("⚖️ Scoring Weights (must sum to 1.0)")
weight_organic = st.sidebar.slider("Weight: Organic", 0.0, 1.0, 0.45, 0.01)
weight_humane = st.sidebar.slider("Weight: Humane", 0.0, 1.0, 0.35, 0.01)
weight_engagement = st.sidebar.slider("Weight: Engagement", 0.0, 1.0, 0.05, 0.01)
weight_similarity = st.sidebar.slider("Weight: (1 - Similarity)", 0.0, 1.0, 0.15, 0.01)



total_weight = round(weight_organic + weight_humane + weight_engagement + weight_similarity, 4)

if total_weight != 1.0:
    st.sidebar.warning(f"⚠️ Total = {total_weight}. Please adjust weights to sum to 1.0.")

if st.button("🔁 Generate Feed") and total_weight == 1.0:
    with st.spinner("Fetching and scoring posts..."):
        feed = generate_feed(
            organicThreshold,
            humane_scoreThreshold,
            engagement_scoreThreshold,
            postSimilarityScoreThreshold,
            weights={
                "organic": weight_organic,
                "humane": weight_humane,
                "engagement": weight_engagement,
                "similarity": weight_similarity
            },
            isCrypto=isCrypto,
            isWomen=isWomen,
            isW30Feed=isW30Feed,
            topicsInput=topicsInput,
            isCreator = isCreator
        )

        st.success(f"✅ Generated {len(feed)} posts!")

        for post in feed:
            st.markdown(f"**User ID**: `{post['handle']}` [🔗 View on Firefly]({post['fireflyLink']})  \n  **Hash**: `{post['hash']}`  \n**CreatorScore**: `{post['score']}` \n**Similarity**: `{post['similarity']}` \n\n**Post**: {post['text']}")
            st.markdown("---")
