from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from typing import Dict,List,Optional,Annotated
from rapidfuzz import process,fuzz
import pandas as pd
import joblib
from contextlib import asynccontextmanager

app=FastAPI()


artifacts = {} # artifacts are basically the files of ML model or data preprocessing file made for later use iin this case the scaler.pkl the knn_model.pkl etc are artifacts


@asynccontextmanager
async def life_span(app:FastAPI):
    df=pd.read_csv(r"C:\Users\abdul\OneDrive\Desktop\FAST_APPLICATIONS\anime_recommender\data\encoded_anime_data_new.csv")
    anime_title=df["Title"].tolist()  # making an array of just the titles in dataset
    image_links_df=pd.read_csv(r"C:\Users\abdul\OneDrive\Desktop\FAST_APPLICATIONS\anime_recommender\data\cleaned_data.csv")
    image_links=image_links_df["image_link"]
    kmeans=joblib.load(r"C:\Users\abdul\OneDrive\Desktop\FAST_APPLICATIONS\anime_recommender\models\kmeans_model.pkl")
    knn=joblib.load(r"C:\Users\abdul\OneDrive\Desktop\FAST_APPLICATIONS\anime_recommender\models\knn_model.pkl")
    scaler=joblib.load(r"C:\Users\abdul\OneDrive\Desktop\FAST_APPLICATIONS\anime_recommender\models\scaler.pkl")

    
    artifacts["df"]=df
    artifacts["titles"]=anime_title
    artifacts["scaler"]=scaler
    artifacts["knn"]=knn
    artifacts["kmeans"]=kmeans

    non_feature_cols=["Title","Cluster","image_link"]
    feature_cols=[c for c in df.columns if c not in non_feature_cols]
    artifacts["feature_cols"]=feature_cols
    artifacts["feature_matrix"]=df[feature_cols].values

    yield
    artifacts.clear()


app=FastAPI(lifespan=life_span)



def fuzzy_find_title(query:str,limit:int=5,threshold: int=80):
    titles=artifacts["titles"]
    results=process.extract(query,titles,scorer=fuzz.WRatio,limit=limit) #wratio will compare the input string with all the anime titles,limit tells how many to pick threshold tells percentage of similarity in animes titles
    
    matches=[]
    for title,score,index in results:
        matches.append({"title":title,"score":score,"index":index})
    return matches


# # --- Request & Response Schemas ---

class SearchInput(BaseModel):
    query:str   # makng a query variable string type as user will enter a string 
    limit:Optional[int]=Field(default=5,ge=1,le=10)
    
    threshold:Optional[int]=Field(default=85,ge=80,le=100)

class RecommendInput(BaseModel):
    query:str
    top_n:Optional[int]=Field(default=5, ge=1, le=20)
    score_threshold: Optional[int] = Field(default=80, ge=0, le=100)

class RecommendAnime(BaseModel):
    title:str
    distance:float
    cluster:int


class RecommendResponse(BaseModel):
    matched_title:str  
    matched_score:float  
    recommendations:list[RecommendAnime]  



@app.get("/")
def home():
    return f"Welcome to home"




@app.post("/search")
def search(req:SearchInput): # as the input this function is getting will be an object of the SearchInput class and coming from there
    matches=fuzzy_find_title(req.query)
    if not matches:
        return {"message":"No anime found. perhaps the spelling are wrong?"}
    return {"matches":matches}


@app.post("/recommend",response_model=RecommendResponse)
def get_recommend(req:RecommendInput):
#     """
#     1. Fuzzy matches input title against the dataset.
#     2. Takes closest match feature vector.
#     3. Runs KNN model to fetch closest recommendedList.
#     """
    df = artifacts["df"]
    scaler = artifacts["scaler"]
    knn = artifacts["knn"]
    kmeans = artifacts["kmeans"]
    features = artifacts["feature_matrix"]

#     # Step 1: Resolve query string to nearest matching anime title

    best_match = process.extractOne(
        req.query, 
        artifacts["titles"], 
        scorer=fuzz.WRatio
    )
    if not best_match :#or best_match[1] < req.score_threshold:
        raise HTTPException(
                status_code=404,
                detail=f"Could not find a confident match for '{req.query}'. Try adjusting your search query."
            )

    matched_title, match_score, matched_idx = best_match[0], best_match[1], best_match[2]
#     # Step 2: Extract the target feature vector and apply scaling if unscaled
    target_vector = features[matched_idx].reshape(1, -1)
    
    # If feature_matrix in CSV wasn't pre-scaled before saving, scale it here:
    # scaled_vector = scaler.transform(target_vector)
    # Otherwise, pass target_vector directly to KNN:
    scaled_vector = target_vector 
    

   # Step 3: Fetch top K Nearest Neighbors (top_n + 1 to exclude input anime itself)
    distances, indices = knn.kneighbors(scaled_vector, n_neighbors=req.top_n + 1)

    recommendations = []
    for dist, idx in zip(distances[0], indices[0]):
        # Skip the exact matched query anime itself
        if idx == matched_idx:
            continue

        rec_title = df.iloc[idx]["Title"]
        rec_vector = features[idx].reshape(1, -1)
        rec_cluster = int(kmeans.predict(rec_vector)[0]) if kmeans else 0

        recommendations.append(RecommendAnime(
            title=rec_title,distance=round(float(dist),4),
            cluster=rec_cluster
        ))
        if len(recommendations)==req.top_n:
            break
    return RecommendResponse(
        matched_title=matched_title,
        matched_score=round(float(match_score), 2)
        ,recommendations=recommendations
    )



# from contextlib import asynccontextmanager
# from fastapi import FastAPI, HTTPException, Query
# from pydantic import BaseModel, Field
# from rapidfuzz import process, fuzz
# import pandas as pd
# import joblib
# import numpy as np

# # Global dictionary to store loaded artifacts in memory
# artifacts = {} # artifacts are basically the files of ML model or data preprocessing file made for later use iin this case the scaler.pkl the knn_model.pkl etc are artifacts

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # 1. Load dataset & model files on server startup
#     df = pd.read_csv(r"C:\Users\abdul\OneDrive\Desktop\FAST_APPLICATIONS\anime_recommender\data\encoded_anime_data_new.csv")
#     scaler = joblib.load(r"C:\Users\abdul\OneDrive\Desktop\FAST_APPLICATIONS\anime_recommender\models\scaler.pkl")
#     knn = joblib.load(r"C:\Users\abdul\OneDrive\Desktop\FAST_APPLICATIONS\anime_recommender\models\knn_model.pkl")
#     kmeans = joblib.load(r"C:\Users\abdul\OneDrive\Desktop\FAST_APPLICATIONS\anime_recommender\models\kmeans_model.pkl")

#     # Store full DataFrame and titles list for RapidFuzz
#     artifacts["df"] = df
#     artifacts["titles"] = df["Title"].tolist()
#     artifacts["scaler"] = scaler
#     artifacts["knn"] = knn
#     artifacts["kmeans"] = kmeans

#     # 2. Extract feature matrix (exclude non-feature metadata columns)
#     # Adjust non-feature column names according to your dataset
#     non_feature_cols = ["Title", "Cluster", "image_link"]
#     feature_cols = [c for c in df.columns if c not in non_feature_cols]
#     artifacts["feature_cols"] = feature_cols
#     artifacts["feature_matrix"] = df[feature_cols].values

#     yield
#     artifacts.clear()

# app = FastAPI(title="Fuzzy-Matching Anime Recommendation API", lifespan=lifespan)


# # --- Request & Response Schemas ---

# class SearchInput(BaseModel):
#     query: str
#     limit: Optional[int] = Field(default=5, ge=1, le=20)
#     threshold: Optional[int] = Field(default=70, ge=0, le=100)

# class RecommendInput(BaseModel):
#     query: str
#     top_n: Optional[int] = Field(default=5, ge=1, le=20)
#     score_threshold: Optional[int] = Field(default=60, ge=0, le=100)

# class RecommendedAnime(BaseModel):
#     title: str
#     distance: float
#     cluster: int

# class RecommendResponse(BaseModel):
#     matched_title: str
#     match_score: float
#     recommendations: list[RecommendedAnime]


# # --- Helper Search Functions ---

# def fuzzy_find_titles(query: str, limit: int = 5, threshold: int = 70):
#     """Returns top fuzzy matched titles along with their index and similarity score."""
#     titles = artifacts["titles"]
#     results = process.extract(
#         query, 
#         titles, 
#         scorer=fuzz.WRatio, 
#         limit=limit
#     )
    
#     matches = []
#     for title, score, index in results:
#         if score >= threshold:
#             matches.append({"index": index, "title": title, "score": score})
#     return matches


# # --- Endpoints ---

# @app.post("/search")
# def search_titles(req: SearchInput):
#     """Endpoint for frontend search autocomplete and typo-tolerant search."""
#     matches = fuzzy_find_titles(req.query, limit=req.limit, threshold=req.threshold)
#     if not matches:
#         return {"message": "No anime found. Check the spelling or lower the threshold."}
#     return {"matches": matches}


# @app.post("/recommend", response_model=RecommendResponse)
# def get_recommendations(req: RecommendInput):
#     """
#     1. Fuzzy matches input title against the dataset.
#     2. Takes closest match feature vector.
#     3. Runs KNN model to fetch closest recommendations.
#     """
#     df = artifacts["df"]
#     scaler = artifacts["scaler"]
#     knn = artifacts["knn"]
#     kmeans = artifacts["kmeans"]
#     features = artifacts["feature_matrix"]

#     # Step 1: Resolve query string to nearest matching anime title
#     best_match = process.extractOne(
#         req.query, 
#         artifacts["titles"], 
#         scorer=fuzz.WRatio
#     )

#     if not best_match or best_match[1] < req.score_threshold:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Could not find a confident match for '{req.query}'. Try adjusting your search query."
#         )

#     matched_title, match_score, matched_idx = best_match[0], best_match[1], best_match[2]

#     # Step 2: Extract the target feature vector and apply scaling if unscaled
#     target_vector = features[matched_idx].reshape(1, -1)
    
#     # If feature_matrix in CSV wasn't pre-scaled before saving, scale it here:
#     # scaled_vector = scaler.transform(target_vector)
#     # Otherwise, pass target_vector directly to KNN:
#     scaled_vector = target_vector 

#     # Step 3: Fetch top K Nearest Neighbors (top_n + 1 to exclude input anime itself)
#     distances, indices = knn.kneighbors(scaled_vector, n_neighbors=req.top_n + 1)

#     recommendations = []
#     for dist, idx in zip(distances[0], indices[0]):
#         # Skip the exact matched query anime itself
#         if idx == matched_idx:
#             continue

#         rec_title = df.iloc[idx]["Title"]
#         rec_vector = features[idx].reshape(1, -1)
#         rec_cluster = int(kmeans.predict(rec_vector)[0]) if kmeans else 0

#         recommendations.append(
#             RecommendedAnime(
#                 title=rec_title,
#                 distance=round(float(dist), 4),
#                 cluster=rec_cluster
#             )
#         )
        
#         if len(recommendations) == req.top_n:
#             break

#     return RecommendResponse(
#         matched_title=matched_title,
#         match_score=round(float(match_score), 2),
#         recommendations=recommendations
#     )