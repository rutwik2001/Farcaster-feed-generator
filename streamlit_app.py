import streamlit as st
from pymongo import MongoClient
from operator import itemgetter
import time
import math

# === CONFIGURATION ===
MONGO_URI = st.secrets["MONGO_URI"]
DB_NAME = "Social_Media"
USER_COLLECTION = "farcaster_users"
POST_COLLECTION = "farcaster_new_casts"
TOP_N_CREATORS = 200
POSTS_PER_CREATOR = 1
MIN_CREATOR_SCORE = 0.5

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
def generate_feed(organicThreshold, humane_scoreThreshold, engagement_scoreThreshold, postSimilarityScoreThreshold, weights):
    now = int(time.time() * 1000)
    past_24h = now - (2 * 24 * 60 * 60 * 1000)

    recent_user_ids = posts_col.distinct("userId", {
        "timeInUnix": {"$gte": past_24h}
    })

    filtered_users = users_col.find({
        "userId": {"$in": recent_user_ids},
        "authenticityScore": {"$exists": True},
        "organicScore": {"$gt": organicThreshold},
        "humane_score": {"$gt": humane_scoreThreshold},
        "engagement_score": {"$gt": engagement_scoreThreshold},
        "postSimilarityScore": {"$lt": postSimilarityScoreThreshold}
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
            "displayName": user["displayName"],
            "userName": user["username"],
            "creator_score": score
        })
        user_dict[user["userId"]] = score
        user_link[user["userId"]] = f"https://firefly.social/profile/farcaster/{user["username"]}/feed"

    top_creators = sorted(user_scores, key=itemgetter("creator_score"), reverse=True)[:top_n_creators]

    feed_posts = []
    for creator in top_creators:
        posts = posts_col.find({
            "userId": creator["userId"],
            "timeInUnix": {"$gte": past_24h}
        }).sort("timeInUnix", -1).limit(POSTS_PER_CREATOR)

        feed_posts.extend(list(posts))
    filtered_feed = [
        {
            "userId": post["userId"],
            "fireflyLink": user_link.get(post["userId"], ""),
            "hash": post["hash"],
            "text": post.get("text", ""),
            "score": user_dict.get(post["userId"], 0)
        }
        for post in feed_posts if user_dict.get(post["userId"], 0) >= MIN_CREATOR_SCORE and post.get("text", "")
    ]

    return filtered_feed

# === Streamlit Frontend ===
st.title("📡 Farcaster Feed Generator")


st.sidebar.header("📈 Number of users (As number increases time to generate increases)")
top_n_creators = st.sidebar.number_input("Top N Creators", min_value=1, max_value=500, value=100, step=1)

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
            }
        )
        st.success(f"✅ Generated {len(feed)} posts!")

        for post in feed:
            st.markdown(f"**User ID**: `{post['userId']}` [🔗 View on Firefly]({post['fireflyLink']})  \n  **Hash**: `{post['hash']}`  \n**Score**: `{post['score']}`  \n**Post**: {post['text']}")
            st.markdown("---")
