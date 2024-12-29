from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from agentic_rag import create_graph, initialize_agent, logger, get_recommendations_for_request
from langchain_core.messages import HumanMessage
import uuid

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic models
class Senses(BaseModel):
    auditory: Optional[str] = None
    olfactory: Optional[str] = None
    tactile: Optional[str] = None
    visual: Optional[str] = None

class RecommendationRequest(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    behaviours: Optional[List[str]] = None
    interests: Optional[List[str]] = None
    senses: Optional[Senses] = None
    therapy: Optional[List[str]] = None

class Recommendation(BaseModel):
    category: str
    priority: int
    recommendation: str
    reason: str

class RecommendationResponse(BaseModel):
    recommendations: List[Recommendation]

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    try:
        logger.info("Initializing agent during startup")
        await initialize_agent()
        logger.info("Agent initialization complete")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {str(e)}")
        raise

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest, req: Request):
    request_id = str(uuid.uuid4())
    logger.info(f"Starting recommendation request {request_id}")
    
    try:
        # Convert request to dict
        request_data = request.dict()
        logger.info(f"Request {request_id} - Input: {request_data}")
        
        # Get recommendations using utility function
        recommendations = await get_recommendations_for_request(request_data)
        
        logger.info(f"Request {request_id} - Completed successfully")
        return RecommendationResponse(recommendations=recommendations)
        
    except Exception as e:
        logger.error(f"Request {request_id} - Error: {str(e)}", exc_info=True)
        return RecommendationResponse(
            recommendations=[
                Recommendation(
                    category="Behavioral",
                    priority=1,
                    recommendation="Use collaborative problem-solving techniques",
                    reason="Helps reduce frustration and improve communication"
                )
            ]
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7070)