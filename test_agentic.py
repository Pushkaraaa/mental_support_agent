from agentic_rag import (
    create_graph, 
    initialize_agent, 
    setup_logging,
    RecommendationRequest,
    Senses,
    logger
)
from langchain_core.messages import HumanMessage

def test_recommendation(request_data: dict):
    """Test the agentic RAG system with a sample request"""
    try:
        # Initialize logging and agent
        logger.info("Starting test with request data")
        logger.info(f"Input data: {request_data}")
        
        # Convert dict to RecommendationRequest
        request = RecommendationRequest(**request_data)
        
        # Initialize agent and create graph
        initialize_agent()
        graph = create_graph()
        
        # Prepare initial state
        initial_state = {
            "messages": [
                HumanMessage(content="Provide mental health support recommendations")
            ],
            "request_context": request
        }
        
        # Execute graph and collect responses
        logger.info("Starting graph execution")
        final_response = None
        for output in graph.stream(initial_state):
            for key, value in output.items():
                logger.info(f"Output from node '{key}':")
                logger.info("---")
                logger.info(value)
                if key == "convert":
                    final_response = value["messages"][0]
                logger.info("---")
        
        logger.info("Final recommendations:")
        logger.info(final_response)
        return final_response
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Sample test data
    test_data = {
        "name": "John",
        "age": 25,
        "behaviours": ["anxiety", "social withdrawal"],
        "interests": ["music", "reading"],
        "senses": {
            "auditory": "sensitive to loud noises",
            "visual": "prefers dim lighting"
        },
        "therapy": ["CBT"]
    }
    
    # Run test
    result = test_recommendation(test_data)
    
    # Print results in a readable format
    if result and "recommendations" in result:
        print("\nGenerated Recommendations:")
        print("------------------------")
        for i, rec in enumerate(result["recommendations"], 1):
            print(f"\n{i}. {rec['category']} (Priority: {rec['priority']})")
            print(f"Recommendation: {rec['recommendation']}")
            print(f"Reason: {rec['reason']}") 