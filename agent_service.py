from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import httpx
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from agentic_rag import AgentState
import uuid
from pathlib import Path
from logging.handlers import RotatingFileHandler

app = FastAPI()

def setup_logging():
    """Configure logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / 'agent_service.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,
                backupCount=5,
                encoding='utf-8'
            ),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Models (keep your existing models)
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

async def query_retriever(query: str) -> List[str]:
    """Query the retriever service"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/search",
            json={"query": query, "k": 5}
        )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        results = response.json()
        return [doc["content"] for doc in results["documents"]]

@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    request_id = str(uuid.uuid4())
    logger.info(f"Processing request {request_id}")
    
    try:
        # Create context-aware query
        query = f"""
        Generate recommendations for a person with:
        Name: {request.name}
        Age: {request.age}
        Behaviors: {', '.join(request.behaviours or [])}
        Interests: {', '.join(request.interests or [])}
        Therapy: {', '.join(request.therapy or [])}
        """
        logger.info(f"Generated query: {query}")
        
        # Get relevant documents
        context_docs = await query_retriever(query)
        logger.info(f"Retrieved {len(context_docs)} relevant documents")
        
        # Generate recommendations
        llm = ChatOpenAI(temperature=0, model="gpt-4")
        response = llm.invoke(
            [HumanMessage(content=f"""
            Based on these documents:
            {' '.join(context_docs)}
            
            And this user context:
            {request.dict()}
            
            Provide 3 specific recommendations following this format:
            1. Category: [type]
               Priority: [1-5]
               Recommendation: [specific action]
               Reason: [brief explanation]
            """)]
        )
        
        # Parse recommendations
        recommendations = parse_llm_response(response.content)
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return RecommendationResponse(recommendations=recommendations)
        
    except Exception as e:
        logger.error(f"Error processing request {request_id}: {e}", exc_info=True)
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

def grade_documents(state: AgentState) -> str:
    """Determine next step based on retrieved documents"""
    messages = state["messages"]
    if not messages:
        return "rewrite"
    last_message = messages[-1]
    
    if "no relevant information found" in last_message.content.lower():
        return "rewrite"
    return "generate"

def generate(state: AgentState):
    """Generate the final answer"""
    logger.info("Starting answer generation")
    messages = state["messages"]
    question = messages[0].content
    
    # Get the retrieved documents from the last message
    last_message = messages[-1]
    if not hasattr(last_message, 'content') or not last_message.content:
        logger.error("No documents found in messages")
        return {"messages": ["Error: No context available"]}

    docs = last_message.content
    logger.info(f"Generating answer using context: {docs[:200]}...")

    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    response = llm.invoke(
        [HumanMessage(content=f"""
        Based on this context: {docs}
        
        Generate recommendations for: {question}
        
        Format as structured recommendations with category, priority, recommendation, and reason.
        """)]
    )
    
    return {"messages": [response], "request_context": state.get("request_context")}

def parse_llm_response(response_text: str) -> List[Recommendation]:
    """Parse LLM response into structured recommendations"""
    logger.info("Parsing LLM response")
    
    try:
        llm = ChatOpenAI(temperature=0)
        parser_prompt = PromptTemplate.from_template("""
        Convert this response into structured recommendations:
        {response}
        
        Format each recommendation as:
        {
            "category": "[Behavioral/Educational/Therapeutic/Social/Environmental]",
            "priority": [1-5],
            "recommendation": "specific action",
            "reason": "brief explanation"
        }
        
        Return exactly 3 recommendations in valid JSON format.
        """)
        
        parsed = llm.invoke(
            parser_prompt.format(response=response_text)
        )
        
        import json
        recommendations = json.loads(parsed.content)
        return [Recommendation(**rec) for rec in recommendations]
        
    except Exception as e:
        logger.error(f"Failed to parse response: {e}")
        return [
            Recommendation(
                category="Behavioral",
                priority=1,
                recommendation="Use collaborative problem-solving techniques",
                reason="Helps reduce frustration and improve communication"
            )
        ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7070) 