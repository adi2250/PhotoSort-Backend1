"""
PhotoSort Engine - Backend API
===============================
A high-performance FastAPI service designed for automated photography culling.
Key capabilities:
  1. Computer Vision Quality Scoring (Sharpness via Laplacian variance, Luminance analysis).
  2. Haar Cascade Facial Landmark & Eye-Blink Evaluation.
  3. 64-bit Difference Perceptual Hashing (dHash) for burst duplicate detection.
  4. Dual-tier WebP compressive storage to optimize MongoDB Atlas cluster limits.
  5. Asynchronous CRUD operations with MongoDB Atlas via Motor.
"""

import os
import cv2
import base64
import numpy as np
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from bson import ObjectId
from PIL import Image, ExifTags
import io

# Load environment variables from a local .env file if present
load_dotenv()

# ==============================================================================
# DATABASE CONFIGURATION & ASYNC CLIENT SETUP
# ==============================================================================
# Fallback connection string connects to MongoDB Atlas M0 cluster directly
DEFAULT_MONGO = "mongodb+srv://technglobalhosting_db_user:RhbVhLnehkcVIBa1@miniproject.pg2mobb.mongodb.net/?appName=miniproject"
MONGO_URI = os.getenv("MONGO_URI", DEFAULT_MONGO)
DB_NAME = os.getenv("DB_NAME", "photosort_db")

# Initialize non-blocking asynchronous MongoDB client using Motor
client = AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
collection = db["photos"]

# Initialize FastAPI application instance
app = FastAPI(
    title="PhotoSort AI Engine",
    description="Automated computer vision photo classification and culling engine",
    version="2.0.0"
)

# Configure Cross-Origin Resource Sharing (CORS) to allow requests from
# localhost:8000, Live Server (127.0.0.1:5500), and hosted domains (GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# OPENCV HAAR CASCADE INITIALIZATION (SAFEGUARDED)
# ==============================================================================
# Safely initialize Haar Cascades for facial and eye detection. If OpenCV is installed
# in an environment without pre-bundled XML models, default gracefully to None.
face_cascade = None
eye_cascade = None

if hasattr(cv2, 'CascadeClassifier') and hasattr(cv2, 'data'):
    try:
        face_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        eye_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        
        fc = cv2.CascadeClassifier(face_path)
        ec = cv2.CascadeClassifier(eye_path)
        
        if not fc.empty():
            face_cascade = fc
        if not ec.empty():
            eye_cascade = ec
    except Exception:
        # Fall back to None so the server continues running without crashing
        face_cascade = None
        eye_cascade = None

# ==============================================================================
# REQUEST VALIDATION SCHEMAS (PYDANTIC)
# ==============================================================================
class ClassificationUpdate(BaseModel):
    """Schema for updating a single image's classification tag."""
    category: str = Field(..., pattern="^(good|review|poor)$", description="Target category: good, review, or poor")

class BatchCategoryUpdate(BaseModel):
    """Schema for updating multiple image categories in a single bulk operation."""
    ids: List[str] = Field(..., description="List of MongoDB ObjectId hex strings")
    category: str = Field(..., pattern="^(good|review|poor)$", description="Target category: good, review, or poor")

class BatchDeleteRequest(BaseModel):
    """Schema for deleting multiple images in a single bulk operation."""
    ids: List[str] = Field(..., description="List of MongoDB ObjectId hex strings to delete")

# ==============================================================================
# COMPUTER VISION & UTILITY HELPER FUNCTIONS
# ==============================================================================
def compute_dhash(img_bgr: np.ndarray) -> str:
    """
    Calculates a 64-bit Difference Hash (dHash) for perceptual image comparison.
    1. Resizes image to 9x8 grayscale (72 pixels).
    2. Compares adjacent horizontal pixels (8 differences across 8 rows = 64 bits).
    3. Returns a 16-character hexadecimal string representing the visual fingerprint.
    """
    resized = cv2.resize(img_bgr, (9, 8), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]
    
    hash_int = 0
    for bit in diff.flatten():
        hash_int = (hash_int << 1) | int(bit)
    return f"{hash_int:016x}"

def hamming_distance(h1: str, h2: str) -> int:
    """
    Calculates the bitwise Hamming distance between two hex hashes.
    A distance of 0 indicates an exact visual duplicate.
    A distance <= 8 indicates near-identical burst sequences or slight camera movements.
    """
    return bin(int(h1, 16) ^ int(h2, 16)).count('1')

def extract_exif_data(raw_bytes: bytes) -> dict:
    """
    Extracts camera hardware metadata (Camera model, Lens, ISO, F-number, Shutter speed)
    from raw image bytes using PIL.
    """
    meta = {
        "camera": "Standard Camera",
        "lens": "Standard Lens",
        "iso": "Auto",
        "f_stop": "N/A",
        "shutter": "N/A"
    }
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        exif = pil_img.getexif()
        if exif:
            make = exif.get(ExifTags.Base.Make, "")
            model = exif.get(ExifTags.Base.Model, "")
            if make or model:
                meta["camera"] = f"{make} {model}".strip()
            iso_val = exif.get(ExifTags.Base.ISOSpeedRatings)
            if iso_val:
                meta["iso"] = f"ISO {iso_val}"
            f_num = exif.get(ExifTags.Base.FNumber)
            if f_num:
                meta["f_stop"] = f"f/{float(f_num):.1f}"
            exp_time = exif.get(ExifTags.Base.ExposureTime)
            if exp_time:
                meta["shutter"] = f"{exp_time}s"
    except Exception:
        pass
    return meta

def compress_to_webp(img_bgr: np.ndarray, max_dim: int = 1600, quality: int = 80) -> str:
    """
    Reduces memory footprint by resizing large images and encoding them into WebP format.
    Returns a standard base64 Data URL ready for frontend rendering and MongoDB storage.
    """
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    
    success, encoded = cv2.imencode('.webp', img_bgr, [cv2.IMWRITE_WEBP_QUALITY, quality])
    if not success:
        raise ValueError("WebP compression failed")
    
    b64 = base64.b64encode(encoded).decode("utf-8")
    return f"data:image/webp;base64,{b64}"

def analyze_image_cv(image_bytes: bytes):
    """
    Primary Computer Vision Quality Assessment Pipeline:
      1. Decodes raw bytes to OpenCV BGR matrix.
      2. Computes Laplacian Variance (sharpness/blur detection).
      3. Analyzes 8-bit Luminance histogram (detects underexposure / blown highlights).
      4. Detects frontal faces and checks eye aspect regions for blinks/closed eyes.
      5. Generates 64-bit dHash for duplicate identification.
      6. Synthesizes an overall Quality Score (10-99) and classification category.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Invalid image file or unsupported format.")

    # Convert to grayscale for gradient and luminance math
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Blur Detection via Laplacian Second-Derivative Kernel
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = round(float(laplacian_var), 2)
    
    # 2. Exposure Analysis via Mean Luminance
    mean_lum = round(float(np.mean(gray)), 2)

    # 3. Facial Analysis & Closed Eye / Blink Detection
    face_count = 0
    eyes_closed = False

    if face_cascade is not None:
        try:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))
            face_count = int(len(faces))

            # If faces exist, inspect eye regions inside each bounding box
            if face_count > 0 and eye_cascade is not None:
                for (fx, fy, fw, fh) in faces:
                    roi_gray = gray[fy:fy + fh, fx:fx + fw]
                    eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=4, minSize=(15, 15))
                    if len(eyes) == 0:
                        eyes_closed = True
                        break
        except Exception:
            pass

    issues = []
    score = 92  # Base quality benchmark

    # Penalize out of focus / soft images
    if sharpness_score < 60.0:
        issues.append("Out of Focus")
        score -= 40
    elif sharpness_score < 120.0:
        issues.append("Soft Focus")
        score -= 15

    # Penalize deep underexposure or blown specular highlights
    if mean_lum < 45.0:
        issues.append("Underexposed")
        score -= 30
    elif mean_lum > 215.0:
        issues.append("Blown Highlights")
        score -= 30

    # Penalize subjects blinking during capture
    if eyes_closed:
        issues.append("Closed Eyes / Blink")
        score -= 20

    # Clamp final score between 10 and 99
    score = max(10, min(99, int(score)))

    # Assign culling bucket based on score thresholds
    if score >= 75:
        category = "good"
    elif score >= 50:
        category = "review"
    else:
        category = "poor"

    if not issues:
        issues = ["Sharp Focus", "Balanced Exposure"]

    # 4. Generate visual difference hash
    dhash = compute_dhash(img)

    return {
        "score": score,
        "category": category,
        "sharpness": sharpness_score,
        "exposure": mean_lum,
        "issues": issues,
        "face_count": face_count,
        "eyes_closed": eyes_closed,
        "resolution": f"{img.shape[1]}x{img.shape[0]}",
        "dhash": dhash,
        "img_cv": img
    }

def format_doc(doc):
    """Converts MongoDB BSON ObjectId into a JSON-serializable string id."""
    doc["id"] = str(doc["_id"])
    del doc["_id"]
    return doc

# ==============================================================================
# FASTAPI ENDPOINTS & ROUTING
# ==============================================================================

@app.get("/")
async def serve_frontend():
    """Serves the static index.html file for root browser navigation."""
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "index.html not found in root directory"}

@app.get("/api/health")
async def health_check():
    """Pings MongoDB Atlas cluster to verify connection status."""
    try:
        await client.admin.command('ping')
        db_status = "connected"
    except Exception as e:
        db_status = f"unreachable: {str(e)}"
    return {"status": "ok", "mongodb": db_status}

@app.post("/api/photos/analyze")
async def analyze_and_save_photo(file: UploadFile = File(...)):
    """
    Uploads a photo, runs computer vision analysis, compresses images to dual-tier WebP,
    checks for burst duplicates against recent database records, and saves the record.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    content = await file.read()
    
    # Check max file threshold (15MB to adhere to MongoDB BSON limits)
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max size is 15MB.")

    try:
        cv_results = analyze_image_cv(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Image processing failed: {str(e)}")

    # Extract camera metadata
    exif_meta = extract_exif_data(content)
    
    # Compress image into dual-tier representations:
    # 1. Display WebP (1600px max dimension) for full-screen inspection
    # 2. Thumbnail WebP (400px max dimension) for the workspace grid
    img_cv = cv_results.pop("img_cv")
    compressed_image = compress_to_webp(img_cv, max_dim=1600, quality=82)
    thumbnail_image = compress_to_webp(img_cv, max_dim=400, quality=70)

    # Check for burst duplicates against the 50 most recent uploads
    dhash = cv_results["dhash"]
    is_duplicate = False
    duplicate_of = None

    recent_photos = collection.find({}, {"dhash": 1, "filename": 1, "score": 1}).sort("created_at", -1).limit(50)
    async for p in recent_photos:
        if "dhash" in p:
            dist = hamming_distance(dhash, p["dhash"])
            if dist <= 8:  # Visual threshold for burst sequences
                is_duplicate = True
                duplicate_of = str(p["_id"])
                cv_results["issues"].append("Burst Duplicate")
                break

    # Build the document to persist in MongoDB
    record = {
        "filename": file.filename,
        "image_data": compressed_image,
        "thumb_data": thumbnail_image,
        "score": cv_results["score"],
        "category": cv_results["category"],
        "sharpness": cv_results["sharpness"],
        "exposure": cv_results["exposure"],
        "issues": cv_results["issues"],
        "resolution": cv_results["resolution"],
        "face_count": cv_results["face_count"],
        "eyes_closed": cv_results["eyes_closed"],
        "dhash": dhash,
        "is_duplicate": is_duplicate,
        "duplicate_of": duplicate_of,
        "exif": exif_meta,
        "file_size_kb": round(len(content) / 1024, 2),
        "compressed_size_kb": round(len(compressed_image) * 0.75 / 1024, 2),
        "created_at": datetime.utcnow()
    }

    result = await collection.insert_one(record)
    record["id"] = str(result.inserted_id)
    if "_id" in record:
        del record["_id"]

    return record

@app.get("/api/photos")
async def get_photos(
    category: Optional[str] = Query(None, pattern="^(good|review|poor)$"),
    sort: Optional[str] = Query("newest")
):
    """
    Retrieves stored photos from MongoDB Atlas.
    Supports optional category filtering and sorting by date, score, sharpness, or file size.
    """
    query = {}
    if category:
        query["category"] = category

    sort_field = "created_at"
    sort_dir = -1

    if sort == "oldest":
        sort_dir = 1
    elif sort == "score_desc":
        sort_field = "score"
    elif sort == "score_asc":
        sort_field = "score"
        sort_dir = 1
    elif sort == "sharpness_desc":
        sort_field = "sharpness"
    elif sort == "size_desc":
        sort_field = "file_size_kb"

    cursor = collection.find(query).sort(sort_field, sort_dir)
    photos = []
    async for doc in cursor:
        photos.append(format_doc(doc))
    return photos

@app.patch("/api/photos/{photo_id}/category")
async def update_classification(photo_id: str, payload: ClassificationUpdate):
    """Updates the classification category (good/review/poor) of an individual photo."""
    if not ObjectId.is_valid(photo_id):
        raise HTTPException(status_code=400, detail="Invalid photo ID format.")

    result = await collection.update_one(
        {"_id": ObjectId(photo_id)},
        {"$set": {"category": payload.category}}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Photo record not found.")

    return {"status": "success", "id": photo_id, "category": payload.category}

@app.post("/api/photos/batch-category")
async def batch_update_category(payload: BatchCategoryUpdate):
    """Applies a classification category to multiple selected photos simultaneously."""
    valid_ids = [ObjectId(pid) for pid in payload.ids if ObjectId.is_valid(pid)]
    if not valid_ids:
        raise HTTPException(status_code=400, detail="No valid IDs provided.")

    result = await collection.update_many(
        {"_id": {"$in": valid_ids}},
        {"$set": {"category": payload.category}}
    )

    return {"status": "success", "modified_count": result.modified_count}

@app.delete("/api/photos/{photo_id}")
async def delete_photo(photo_id: str):
    """Permanently deletes an individual photo record from MongoDB Atlas."""
    if not ObjectId.is_valid(photo_id):
        raise HTTPException(status_code=400, detail="Invalid photo ID format.")

    result = await collection.delete_one({"_id": ObjectId(photo_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Photo record not found.")

    return {"status": "success", "message": "Photo deleted successfully", "id": photo_id}

@app.post("/api/photos/batch-delete")
async def batch_delete_photos(payload: BatchDeleteRequest):
    """Permanently deletes multiple selected photo records from MongoDB Atlas in one operation."""
    valid_ids = [ObjectId(pid) for pid in payload.ids if ObjectId.is_valid(pid)]
    if not valid_ids:
        raise HTTPException(status_code=400, detail="No valid IDs provided.")

    result = await collection.delete_many({"_id": {"$in": valid_ids}})
    return {"status": "success", "deleted_count": result.deleted_count}

@app.get("/api/stats")
async def get_database_stats():
    """
    Computes storage quota usage and quality analytics:
      1. Executes dbStats command on MongoDB Atlas to compute exact storage in MB.
      2. Calculates remaining MB based on the 512 MB Free Tier quota limit.
      3. Aggregates average quality scores, sharpness, exposure, and category counts.
    """
    try:
        db_stats = await db.command("dbStats")
        storage_bytes = db_stats.get("storageSize", db_stats.get("dataSize", 0))
        data_bytes = db_stats.get("dataSize", 0)
    except Exception:
        storage_bytes = 0
        data_bytes = 0

    # Atlas M0 Free Tier storage ceiling
    LIMIT_MB = 512.0
    used_bytes = max(storage_bytes, data_bytes)
    used_mb = round(used_bytes / (1024 * 1024), 2)
    remaining_mb = round(max(0.0, LIMIT_MB - used_mb), 2)
    used_percent = round((used_mb / LIMIT_MB) * 100, 2)

    # Document counting queries
    total_photos = await collection.count_documents({})
    good_count = await collection.count_documents({"category": "good"})
    review_count = await collection.count_documents({"category": "review"})
    poor_count = await collection.count_documents({"category": "poor"})
    duplicate_count = await collection.count_documents({"is_duplicate": True})

    # MongoDB aggregation pipeline to compute global performance benchmarks
    pipeline = [
        {
            "$group": {
                "_id": None,
                "avg_score": {"$avg": "$score"},
                "avg_sharpness": {"$avg": "$sharpness"},
                "avg_exposure": {"$avg": "$exposure"},
                "avg_file_size_kb": {"$avg": "$file_size_kb"}
            }
        }
    ]
    agg_results = await collection.aggregate(pipeline).to_list(1)
    averages = agg_results[0] if agg_results else {}

    return {
        "storage": {
            "used_mb": used_mb,
            "remaining_mb": remaining_mb,
            "limit_mb": LIMIT_MB,
            "used_percent": used_percent,
            "total_bytes": used_bytes
        },
        "counts": {
            "total": total_photos,
            "good": good_count,
            "review": review_count,
            "poor": poor_count,
            "duplicates": duplicate_count
        },
        "averages": {
            "score": round(averages.get("avg_score") or 0, 1),
            "sharpness": round(averages.get("avg_sharpness") or 0, 1),
            "exposure": round(averages.get("avg_exposure") or 0, 1),
            "file_size_kb": round(averages.get("avg_file_size_kb") or 0, 1)
        }
    }
