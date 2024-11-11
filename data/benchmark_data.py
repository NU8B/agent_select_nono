# Our own benchmark
# Dictionary mapping queries to their ideal agents
QUERY_AGENT_MAPPING = {
    "Suggest a 2-week Italian food tour itinerary.": "Travel Planning Agent",
    "How do I implement a persistent shopping cart in a MERN stack e-commerce site?": "Expert MERN Developer",
    "Design a microservices architecture for a social media platform using Java and Spring Boot.": "Senior Backend Engineer",
    "Guide me through fine-tuning a BERT model for sentiment analysis.": "NLP Specialist",
    "Use pandas and matplotlib to analyze financial data.": "Data Scientist",
    "Implement offline translation in a React Native app using TensorFlow Lite and AsyncStorage.": "Mobile App Developer",
    "Create interactive climate change visualizations with D3.js.": "Data Scientist",
    "Explain how to build a hybrid recommendation system for a streaming platform.": "Systems Architect",
    "What are the latest CRISPR advancements for treating cystic fibrosis?": "Medical Consultant",
    "Plan a cherry blossom season trip to Japan.": "Travel Planning Agent",
    "Help me structure a blog series about my Southeast Asia backpacking trip.": "Traveling Blogger",
    "I'm starting a YouTube channel on quantum physics for a general audience.": "Quantum Physicist",
    "Explain quantum key distribution and the BB84 protocol.": "Quantum Physicist",
    "Guide me through implementing a custom Kubernetes autoscaling solution using Prometheus.": "DevOps Engineer",
    "Design a high-availability microservices architecture for an e-commerce platform using Java Spring Boot and PostgreSQL.": "Senior Backend Engineer",
    "Create a comprehensive CI/CD pipeline for a microservices application using Jenkins, Docker, and Kubernetes.": "DevOps Engineer",
    "Develop a security framework for a financial services API, including OAuth2 implementation, rate limiting, and threat detection.": "Security Expert",
    "Design a cross-platform mobile app for real-time video streaming using React Native.": "Mobile App Developer",
    "Implement a hybrid recommendation system combining collaborative filtering and content-based approaches using Python and TensorFlow.": "AI Expert",
    "Design a real-time data processing pipeline for IoT sensor data using Apache Kafka, Spark Streaming, and MongoDB.": "Data Engineer",
    "Create a multiplayer mobile game using Unity with real-time synchronization, matchmaking, and leader boards.": "Game Developer",
    "Design a comprehensive design system for a financial dashboard application.": "UI/UX Designer",
    "Develop a DeFi lending platform using Solidity and Web3.js.": "Blockchain Developer",
    "Create an adaptive learning platform that personalizes content based on student performance.": "Education Specialist",
    "Explain the basics of blockchain technology and its potential applications beyond cryptocurrency.": "Blockchain Developer",
    "How do I set up a home hydroponics system for growing vegetables?": "Interior Designer",
    "Compare and contrast supervised, unsupervised, and reinforcement learning in AI.": "AI Expert",
    "Design a scalable architecture for a real-time multiplayer mobile game using Unity and Firebase.": "Systems Architect",
    "Explain the concept of epigenetics and its implications for personalized medicine.": "Medical Consultant",
    "Guide me through creating a responsive web application using Vue.js and Tailwind CSS.": "Expert MERN Developer",
    "What are the latest advancements in fusion energy research?": "Quantum Physicist",
    "Outline a training plan for a beginner aiming to run their first marathon in 6 months.": "Fitness Trainer",
    "How can machine learning be applied to improve renewable energy forecasting and grid management?": "AI Expert",
    "Explain the process of fermentation in food preservation.": "Nutrition Consultant",
    "Discuss the ethical implications of using AI in healthcare decision-making.": "AI Ethics Specialist",
    "Design a sustainable tiny house.": "Interior Designer",
    "Guide me through setting up a home automation system using Raspberry Pi and open-source software.": "Embedded Systems Engineer",
    "Design a comprehensive 12-week training program for a marathon.": "Fitness Trainer",
    "Develop a stress management program for corporate professionals.": "Psychology Counselor",
    "Create a career transition plan for a software developer moving into AI/ML.": "Career Counselor",
    "Plan a 3-week photography-focused trip through Southeast Asia.": "Traveling Blogger",
    "Design a sustainable smart home system using renewable energy sources.": "Interior Designer",
    "Create a GDPR compliance framework for a SaaS platform.": "Legal Advisor",
    "Develop a comprehensive digital marketing strategy for a tech startup.": "Marketing Strategist",
    "Design an IoT-based environmental monitoring system using ESP32.": "Embedded Systems Engineer",
    "Implement a multilingual customer support chatbot using BERT and FastAPI.": "NLP Specialist",
    "Create a personal investment strategy combining traditional and crypto assets.": "Financial Advisor",
}


def get_benchmark_metrics(predictions: dict) -> dict:
    """
    Calculate benchmark metrics from predictions

    Args:
        predictions: Dict mapping queries to predicted agents

    Returns:
        Dict containing accuracy metrics
    """
<<<<<<< Updated upstream:data/benchmark_data.py
    total = len(QUERY_AGENT_MAPPING)
=======
    # Get total number of queries
    total = len(QUERY_AGENT_MAPPING)

    # Count for correct predictions
>>>>>>> Stashed changes:data/test_data.py
    correct = sum(
        1
        for query, agent in predictions.items()
        if agent == QUERY_AGENT_MAPPING.get(query)
    )

    # Return the accuracy metrics
    return {
        "total_queries": total,
        "correct_predictions": correct,
        "accuracy": correct / total if total > 0 else 0,
        "incorrect_predictions": total - correct,
    }


def get_detailed_results(predictions: dict) -> list:
    """
    Get detailed results for each query

    Args:
        predictions: Dict mapping queries to predicted agents

    Returns:
        List of dicts containing query-level results
    """
<<<<<<< Updated upstream:data/benchmark_data.py
    results = []
    for query, predicted_agent in predictions.items():
        correct_agent = QUERY_AGENT_MAPPING.get(query)
=======
    results = []  # Create empty list to store results

    # Loop for each query and predicted agent
    for query, predicted_agent in predictions.items():
        # Get the correct agent for the query
        correct_agent = QUERY_AGENT_MAPPING.get(query)
        # Append the results to the list
>>>>>>> Stashed changes:data/test_data.py
        results.append(
            {
                "query": query,
                "predicted_agent": predicted_agent,
                "correct_agent": correct_agent,
                "is_correct": predicted_agent == correct_agent,
            }
        )
<<<<<<< Updated upstream:data/benchmark_data.py
    return results
=======

    return results
>>>>>>> Stashed changes:data/test_data.py
