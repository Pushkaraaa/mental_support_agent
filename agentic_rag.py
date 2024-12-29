import os
from typing import Annotated, Literal, Sequence, List, Dict, Any, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain import hub
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph.message import add_messages

import os
import logging
import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
import json
import httpx
from langchain.tools import Tool
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

logger = logging.getLogger(__name__)

# Global variables
retriever_tool = None
logger = logging.getLogger(__name__)

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

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    request_context: Optional[RecommendationRequest]

def setup_logging():
    """Configure logging with a single file"""
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Use a single log file
    log_file = log_dir / 'agentic_rag.log'
    
    # Configure logging with more detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[
            # Use RotatingFileHandler to manage file size
            logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,  # Keep 5 backup files
                encoding='utf-8'
            ),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger

async def setup_retriever():
    """Initialize the retriever that uses the external retriever service"""
    logger.info("Starting retriever setup")
    
    @tool
    async def retrieve_docs(query: str) -> str:
        """Search through the mental health documents to find relevant information."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8000/search",
                    json={
                        "query": query,
                        "k": 5
                    }
                )
                response.raise_for_status()
                
                results = response.json()
                
                # Combine all document contents with their scores
                documents = results["documents"]
                scores = results["scores"]
                
                combined_text = "\n\n".join([
                    f"[Score: {score:.4f}]\n{doc['content']}"
                    for doc, score in zip(documents, scores)
                ])
                
                return combined_text
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return "Error: Could not retrieve relevant documents"

    # Check if retriever service is available
    try:
        async with httpx.AsyncClient() as client:
            health_check = await client.get("http://localhost:8000/health")
            health_check.raise_for_status()
    except Exception as e:
        raise RuntimeError(f"Retriever service is not available: {str(e)}")

    logger.info("Retriever setup complete")
    return Tool(
        name="retrieve_docs",
        description="Search through the mental health documents to find relevant information.",
        func=retrieve_docs
    )

def grade_documents(state) -> Literal["generate", "rewrite"]:
    """Grade document relevance"""

    return "generate"

    logger.info("Starting document grading")
    
    messages = state["messages"]
    question = messages[0].content
    
    # Get the last tool response which contains the retrieved documents
    last_message = messages[-1]
    if not hasattr(last_message, 'content') or not last_message.content:
        logger.warning("No documents found in messages")
        return "rewrite"
    
    docs = last_message.content
    logger.info(f"Grading documents: {docs[:200]}...")

    class grade(BaseModel):
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    llm_with_tool = model.with_structured_output(grade)

    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of retrieved documents to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the documents contain keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the documents are relevant to the question.""",
        input_variables=["context", "question"],
    )

    chain = prompt | llm_with_tool
    scored_result = chain.invoke({"question": question, "context": docs})
    result = "generate" if scored_result.binary_score == "yes" else "rewrite"
    logger.info(f"Grading result: {result}")
    return result

def create_rag_query(state) -> str:
    """Create context-aware RAG query"""
    logger.info("Creating context-aware RAG query")
    
    request_dict = state.get("request_context")
    if not request_dict:
        return state["messages"][0].content
    
    # Convert dict back to RecommendationRequest object
    try:
        if request_dict.get("senses"):
            request_dict["senses"] = Senses(**request_dict["senses"])
        request = RecommendationRequest(**request_dict)
    except Exception as e:
        logger.error(f"Error converting request dict to object: {str(e)}")
        return state["messages"][0].content
    
    # Create base context string
    context_parts = []
    if request.name:
        context_parts.append(f"For a person named {request.name}")
    if request.age:
        context_parts.append(f"aged {request.age}")
    if request.behaviours:
        context_parts.append(f"who exhibits behaviors: {', '.join(request.behaviours)}")
    if request.interests:
        context_parts.append(f"with interests in: {', '.join(request.interests)}")
    if request.therapy:
        context_parts.append(f"currently receiving therapies: {', '.join(request.therapy)}")
    if request.senses:
        sensory_info = [f"{k}: {v}" for k, v in request.senses.model_dump().items() if v]
        if sensory_info:
            context_parts.append(f"with sensory considerations: {', '.join(sensory_info)}")
    
    context = ". ".join(context_parts)
    original_query = state["messages"][0].content
    
    # Use LLM to refine the query
    class RefinedQuery(BaseModel):
        query: str = Field(description="The refined search query")

    model = ChatOpenAI(temperature=0, model="gpt-4")
    refiner = model.with_structured_output(RefinedQuery)
    
    prompt = PromptTemplate(
        template="""Given the following context about a person and an original question, 
        create a detailed search query that will help find relevant information from mental health documents.
        
        Context about the person:
        {context}
        
        Original question:
        {query}
        
        Create a detailed, specific query that combines the original question with the person's context.
        Focus on finding information that would be most relevant to this specific person's situation.""",
        input_variables=["context", "query"]
    )
    
    refined = refiner.invoke(
        prompt.format(context=context, query=original_query)
    )
    
    logger.info(f"Original query: {original_query}")
    logger.info(f"Refined query: {refined.query}")
    
    return refined.query

def agent(state):
    """Agent to decide on retrieval"""
    logger.info("Starting agent processing")
    
    # Create context-aware query
    refined_query = create_rag_query(state)
    messages = [HumanMessage(content=refined_query)]
    
    model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4-turbo")
    model = model.bind_tools([retriever_tool])
    response = model.invoke(messages)
    
    logger.info("Agent processing complete")
    return {"messages": [response], "request_context": state.get("request_context")}

def rewrite(state):
    """Rewrite the query"""
    logger.info("Starting query rewrite")
    messages = state["messages"]
    question = messages[0].content
    logger.info(f"Original question: {question}")

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    model = ChatOpenAI(temperature=0, model="gpt-4o-mini", streaming=True)
    response = model.invoke(msg)
    
    logger.info(f"Rewritten question: {response.content}")
    return {"messages": [response], "request_context": state.get("request_context")}

def generate(state):
    """Generate the final answer"""
    logger.info("Starting answer generation")
    messages = state["messages"]
    question = messages[0].content
    
    # Get the retrieved documents from the last message (tool response)
    last_message = messages[-1]
    if not hasattr(last_message, 'content') or not last_message.content:
        logger.error("No documents found in messages")
        return {"messages": ["Error: No context available for answering the question."]}

    docs = last_message.content
    logger.info(f"Generating answer for: {question}")
    logger.info(f"Using context from tool response: {docs[:200]}...")

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({
        "context": docs,
        "question": question
    })
    
    logger.info(f"Generated response: {response[:100]}...")
    return {"messages": [response], "request_context": state.get("request_context")}

def convert_to_recommendations(state):
    """Convert RAG response to API-compatible recommendations"""
    logger.info("Converting RAG response to recommendations")
    
    messages = state["messages"]
    if not messages or not messages[-1]:
        logger.error("No response to convert")
        return {"messages": state["messages"]}
    
    rag_response = messages[-1]
    if isinstance(rag_response, BaseMessage):
        rag_response = rag_response.content
    
    request_context = state.get("request_context", {})
    
    class APIRecommendation(BaseModel):
        category: str = Field(
            description="Category of recommendation",
            enum=["Behavioral", "Educational", "Therapeutic", "Social", "Environmental"]
        )
        priority: int = Field(description="Priority level", ge=1, le=5)
        recommendation: str = Field(description="The actual recommendation")
        reason: str = Field(description="Reason for the recommendation")

    class RecommendationList(BaseModel):
        recommendations: List[APIRecommendation]

    model = ChatOpenAI(temperature=0, model="gpt-4")
    structured_llm = model.with_structured_output(RecommendationList)

    # Simplified prompt that's more direct
    prompt = PromptTemplate(
        template="""Convert this mental health support response into 2-3 specific recommendations.

Response to convert:
{response}

Context about the person (if available):
{context}

Focus on practical, actionable recommendations that are most relevant to this situation.
Each recommendation must include a category (Behavioral/Educational/Therapeutic/Social/Environmental), 
priority (1-5, where 1 is highest), the specific recommendation, and a brief reason why it would help.""",
        input_variables=["context", "response"]
    )

    try:
        result = structured_llm.invoke(
            prompt.format(
                context=str(request_context),
                response=rag_response
            )
        )
        recommendations = [rec.dict() for rec in result.recommendations]
        logger.info(f"Successfully converted response into {len(recommendations)} recommendations")
        return {"messages": [AIMessage(content=json.dumps({"recommendations": recommendations}))]}
    
    except Exception as e:
        logger.error(f"Error converting response: {str(e)}")
        fallback = {
            "recommendations": [{
                "category": "Environmental",
                "priority": 1,
                "recommendation": "Create a calm sensory environment",
                "reason": "Helps reduce sensory overload and provides comfort"
            }]
        }
        return {"messages": [AIMessage(content=json.dumps(fallback))]}

async def initialize_agent():
    """Initialize the agent components"""
    global retriever_tool
    try:
        if retriever_tool is None:
            logger.info("Initializing agent components")
            retriever_tool = await setup_retriever()
        return retriever_tool
    except Exception as e:
        logger.error(f"Failed to initialize retriever tool: {str(e)}")
        raise

def create_graph():
    """Create the workflow graph"""
    logger.info("Creating workflow graph")
    
    global retriever_tool
    if retriever_tool is None:
        raise RuntimeError("Retriever tool not initialized")
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent)
    retrieve = ToolNode([retriever_tool])
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate)
    workflow.add_node("convert", convert_to_recommendations)
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate": "generate",
            "rewrite": "rewrite"
        }
    )
    workflow.add_edge("generate", "convert")  # Add conversion step
    workflow.add_edge("convert", END)  # Conversion leads to end
    workflow.add_edge("rewrite", "agent")
    
    logger.info("Graph compilation complete")
    return workflow.compile()

def save_graph_visualization(graph, filename="graph_visualization.png"):
    """Save the graph visualization as a PNG file"""
    try:
        import base64
        from pathlib import Path
        
        logger.info("Generating graph visualization")
        
        # Get the Mermaid graph as PNG bytes
        mermaid_png = graph.get_graph(xray=True).draw_mermaid_png()
        
        # Save to file
        output_path = Path(filename)
        output_path.write_bytes(mermaid_png)
        
        logger.info(f"Graph visualization saved to {output_path.absolute()}")
        return True
        
    except Exception as e:
        logger.warning(f"Could not generate graph visualization: {str(e)}")
        return False

def main():
    global logger
    logger = setup_logging()
    logger.info("Starting Agentic RAG application")
    
    if "OPENAI_API_KEY" not in os.environ:
        logger.info("OpenAI API key not found in environment")
        os.environ["OPENAI_API_KEY"] = input("Enter your OpenAI API key: ")
    
    try:
        global retriever_tool
        retriever_tool = setup_retriever()
        
        graph = create_graph()
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_graph_visualization(graph, f"agentic_rag_graph_{timestamp}.png")
        
        # Example question (can be modified for mental health context)
        question = "How to give sensory comfort to a child with autism"
        logger.info(f"Processing question: {question}")
        
        inputs = {
            "messages": [
                HumanMessage(content=question),
            ],
            "docs": None
        }
        
        logger.info("Starting graph execution")
        final_response = None
        for output in graph.stream(inputs):
            for key, value in output.items():
                logger.info(f"Output from node '{key}':")
                logger.info("---")
                logger.info(value)
                final_response = value
            logger.info("---")
        
        # Log the final structured recommendations
        if final_response and isinstance(final_response, dict):
            logger.info("Final structured recommendations:")
            logger.info(final_response)
        
        logger.info("Processing complete")
        return final_response
    
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

async def get_recommendations_for_request(request_data: dict) -> List[dict]:
    """
    Utility function to get recommendations based on structured request data.
    
    Args:
        request_data (dict): Dictionary containing request parameters (name, age, behaviors, etc.)
        
    Returns:
        List[dict]: List of recommendation dictionaries
        
    Raises:
        ValueError: If recommendations cannot be generated
    """
    logger.info(f"Processing recommendation request with data: {request_data}")
    
    try:
        # Initialize agent if needed
        global retriever_tool
        if retriever_tool is None:
            retriever_tool = await initialize_agent()
        
        # Create graph
        graph = create_graph()
        
        # Prepare initial state
        initial_state = {
            "messages": [
                HumanMessage(content="Provide mental health support recommendations")
            ],
            "request_context": request_data
        }
        
        # Execute graph and collect final response
        final_response = None
        for output in graph.stream(initial_state):
            for key, value in output.items():
                logger.debug(f"Output from node '{key}': {value}")
                if key == "convert":
                    final_response = value["messages"][0]
        
        if not final_response:
            raise ValueError("No recommendations generated")
        
        # Extract recommendations from AIMessage content
        content = final_response.content
        if isinstance(content, str):
            content = json.loads(content)
        
        recommendations = content.get("recommendations", [])
        logger.info(f"Generated {len(recommendations)} recommendations")
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        # Return fallback recommendations
        return [{
            "category": "Environmental",
            "priority": 1,
            "recommendation": "Create a calm sensory environment",
            "reason": "Helps reduce sensory overload and provides comfort"
        }]

if __name__ == "__main__":
    main()