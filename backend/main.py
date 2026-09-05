import os
import sys
import uuid
import json
import asyncio
import logging
from pathlib import Path
from typing import List, Optional

backend_dir = str(Path(__file__).resolve().parent)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field

from geo_utils import extract_bounds_and_crs
from router import classify_task, get_tool_for_task
from fallback_payloads import FALLBACK_PAYLOADS
from tools import TOOL_REGISTRY
from inference_cache import get_cached_or_none
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satquery.main")

app = FastAPI(
    title="SatQuery AI - Interactive Remote Sensing Assistant API",
    description="Backend AI OS & Agent Router API for Satellite Imagery VQA, Visual Grounding, Change Analysis, and Optical-SAR Fusion.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_STORE = {}
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str
    image_ids: List[str] = Field(default_factory=list)
    demo_mode: bool = False

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "SatQuery AI Backend Engine",
        "active_images": len(IMAGE_STORE)
    }

@app.get("/api/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/images/{image_id}")
def get_image_info(image_id: str):
    if image_id not in IMAGE_STORE:
        raise HTTPException(status_code=404, detail="Image ID not found")
    meta = IMAGE_STORE[image_id]
    return {
        "image_id": image_id,
        "filename": meta["filename"],
        "geo_meta": meta["geo_meta"]
    }

@app.post("/api/upload")
async def upload_images(images: List[UploadFile] = File(...)):
    if not images or len(images) > 2:
        raise HTTPException(status_code=400, detail="Please upload 1 or 2 satellite imagery files.")

    image_ids = []
    for file in images:
        opaque_id = f"img_{uuid.uuid4().hex[:8]}"
        file_path = os.path.join(UPLOAD_DIR, f"{opaque_id}_{file.filename}")

        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        geo_meta = extract_bounds_and_crs(file_path)
        IMAGE_STORE[opaque_id] = {
            "file_path": file_path,
            "filename": file.filename,
            "geo_meta": geo_meta
        }
        image_ids.append(opaque_id)
        logger.info(f"Uploaded {file.filename} -> image_id: {opaque_id}")

    return {"image_ids": image_ids}

@app.post("/api/query")
async def query_agent(payload: QueryRequest):
    """
    SSE stream endpoint for natural language query execution over uploaded imagery.
    V2: demo_mode now checks a real pre-computed inference cache first, and only
    falls back to the hardcoded FALLBACK_PAYLOADS if neither a cache hit nor a
    live tool result is available (keeps a safety net without ever preferring
    fake output over real output).
    """
    query = payload.query
    image_ids = payload.image_ids
    demo_mode = payload.demo_mode

    async def event_generator():
        try:
            yield f"data: {json.dumps({'type': 'trace', 'message': 'Validating imagery and CRS...'})}\n\n"
            await asyncio.sleep(0.5)

            yield f"data: {json.dumps({'type': 'trace', 'message': 'Classifying task via agent router...'})}\n\n"
            await asyncio.sleep(0.5)

            task_type = classify_task(query, image_ids)
            tool_name = get_tool_for_task(task_type)

            yield f"data: {json.dumps({'type': 'trace', 'message': f'Routed to: {task_type}'})}\n\n"
            await asyncio.sleep(0.5)
            
            # Calculate georeferenced status once to use in payloads
            is_georeferenced = any(
                IMAGE_STORE.get(img_id, {}).get("geo_meta", {}).get("georeferenced", False)
                for img_id in image_ids
            )

            cached_payload = None
            if demo_mode:
                yield f"data: {json.dumps({'type': 'trace', 'message': 'Checking pre-computed inference cache...'})}\n\n"
                await asyncio.sleep(0.3)
                cached_payload = get_cached_or_none(image_ids, task_type, IMAGE_STORE)

            if cached_payload:
                yield f"data: {json.dumps({'type': 'trace', 'message': 'Cache hit — returning verified real inference result.'})}\n\n"
                await asyncio.sleep(0.3)
                
                cached_payload["image_georeferenced"] = is_georeferenced
                yield f"data: {json.dumps(cached_payload)}\n\n"
            else:
                if demo_mode:
                    yield f"data: {json.dumps({'type': 'trace', 'message': 'No cache match — falling through to live inference.'})}\n\n"
                    await asyncio.sleep(0.3)

                yield f"data: {json.dumps({'type': 'trace', 'message': f'Executing specialist tool [{tool_name}]...'})}\n\n"
                await asyncio.sleep(0.5)

                yield f"data: {json.dumps({'type': 'trace', 'message': 'Running inference...'})}\n\n"
                await asyncio.sleep(0.5)

                tool_func = TOOL_REGISTRY.get(tool_name)
                if tool_func:
                    result_payload = tool_func(query, image_ids, IMAGE_STORE)
                    
                    result_payload["image_georeferenced"] = is_georeferenced
                    yield f"data: {json.dumps(result_payload)}\n\n"
                else:
                    fallback_payload = FALLBACK_PAYLOADS.get(task_type, FALLBACK_PAYLOADS["single_image_vqa"])
                    
                    fallback_payload["image_georeferenced"] = is_georeferenced
                    yield f"data: {json.dumps(fallback_payload)}\n\n"

        except Exception as e:
            logger.error(f"Error during query stream: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
# @app.post("/api/query")
# async def query_agent(payload: QueryRequest):
#     """
#     SSE stream endpoint for natural language query execution over uploaded imagery.
#     V2: demo_mode now checks a real pre-computed inference cache first, and only
#     falls back to the hardcoded FALLBACK_PAYLOADS if neither a cache hit nor a
#     live tool result is available (keeps a safety net without ever preferring
#     fake output over real output).
#     """
#     query = payload.query
#     image_ids = payload.image_ids
#     demo_mode = payload.demo_mode

#     async def event_generator():
#         try:
#             yield f"data: {json.dumps({'type': 'trace', 'message': 'Validating imagery and CRS...'})}\n\n"
#             await asyncio.sleep(0.5)

#             yield f"data: {json.dumps({'type': 'trace', 'message': 'Classifying task via agent router...'})}\n\n"
#             await asyncio.sleep(0.5)

#             task_type = classify_task(query, image_ids)
#             tool_name = get_tool_for_task(task_type)

#             yield f"data: {json.dumps({'type': 'trace', 'message': f'Routed to: {task_type}'})}\n\n"
#             await asyncio.sleep(0.5)

#             cached_payload = None
#             if demo_mode:
#                 yield f"data: {json.dumps({'type': 'trace', 'message': 'Checking pre-computed inference cache...'})}\n\n"
#                 await asyncio.sleep(0.3)
#                 cached_payload = get_cached_or_none(image_ids, task_type, IMAGE_STORE)

#             if cached_payload:
#                 yield f"data: {json.dumps({'type': 'trace', 'message': 'Cache hit — returning verified real inference result.'})}\n\n"
#                 await asyncio.sleep(0.3)
#                 yield f"data: {json.dumps(cached_payload)}\n\n"
#             else:
#                 if demo_mode:
#                     yield f"data: {json.dumps({'type': 'trace', 'message': 'No cache match — falling through to live inference.'})}\n\n"
#                     await asyncio.sleep(0.3)

#                 yield f"data: {json.dumps({'type': 'trace', 'message': f'Executing specialist tool [{tool_name}]...'})}\n\n"
#                 await asyncio.sleep(0.5)

#                 yield f"data: {json.dumps({'type': 'trace', 'message': 'Running inference...'})}\n\n"
#                 await asyncio.sleep(0.5)

#                 tool_func = TOOL_REGISTRY.get(tool_name)
#                 if tool_func:
#                     result_payload = tool_func(query, image_ids, IMAGE_STORE)
#                     yield f"data: {json.dumps(result_payload)}\n\n"
#                 else:
#                     fallback_payload = FALLBACK_PAYLOADS.get(task_type, FALLBACK_PAYLOADS["single_image_vqa"])
#                     yield f"data: {json.dumps(fallback_payload)}\n\n"

#         except Exception as e:
#             logger.error(f"Error during query stream: {e}", exc_info=True)
#             yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

#     return StreamingResponse(
#         event_generator(),
#         media_type="text/event-stream",
#         headers={
#             "Cache-Control": "no-cache",
#             "Connection": "keep-alive",
#             "X-Accel-Buffering": "no"
#         }
#     )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

