"""
Our Custom Benchmark
Dictionary mapping queries to their ideal agents
"""

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
    "Build a Flask application that scrapes academic papers using BeautifulSoup, processes citations with regex, and provides a REST API with SQLAlchemy for citation network analysis": "Python Developer",
    "Create an adaptive learning platform that personalizes content based on student performance.": "Education Specialist",
    "Explain the basics of blockchain technology and its potential applications beyond cryptocurrency.": "Blockchain Developer",
    "Compare and contrast supervised, unsupervised, and reinforcement learning in AI.": "AI Expert",
    "Design a scalable architecture for a real-time multiplayer mobile game.": "Systems Architect",
    "Explain the concept of epigenetics and its implications for personalized medicine.": "Medical Consultant",
    "Guide me through creating a responsive web application using Vue.js and Tailwind CSS.": "Expert MERN Developer",
    "What are the latest advancements in fusion energy research?": "Quantum Physicist",
    "Outline a training plan for a beginner aiming to run their first marathon in 6 months.": "Fitness Trainer",
    "Create a basic Flask application with pandas for data analysis and simple matplotlib visualizations of CSV files": "Python Engineer",
    "How can machine learning be applied to improve renewable energy forecasting and grid management?": "AI Expert",
    "Create a personalized meal plan for someone with celiac disease and dairy intolerance.": "Nutrition Consultant",
    "Design a sustainable tiny house.": "Interior Designer",
    "Guide me through setting up a home automation system using Raspberry Pi and open-source software.": "Embedded Systems Engineer",
    "Design a comprehensive 12-week training program for a marathon.": "Fitness Trainer",
    "Develop a stress management program for corporate professionals.": "Psychology Counselor",
    "Help me write a Python script to automate log file processing and generate basic test reports": "Python Engineer",
    "Create a career transition plan for a software developer moving into AI/ML.": "Career Counselor",
    "Plan a 3-week photography-focused trip through Southeast Asia.": "Travel Planning Agent",
    "Help me come up with an open-concept living space that maximizes natural light and flow.": "Interior Designer",
    "Provide legal guidance on implementing GDPR requirements and drafting compliance documentation for a SaaS platform, including privacy policies and data processing agreements.": "Legal Advisor",
    "Develop a comprehensive digital marketing strategy for a tech startup.": "Marketing Strategist",
    "Design an IoT-based environmental monitoring system using ESP32.": "Embedded Systems Engineer",
    "Implement a multilingual customer support chatbot using BERT and FastAPI.": "NLP Specialist",
    "Create a personal investment strategy combining traditional and crypto assets.": "Financial Advisor",
    "Develop a multi-cloud disaster recovery solution with active-active configuration.": "Cloud Architect",
    "Design a zero-trust security framework for a multi-cloud enterprise environment.": "Security Expert",
    "Build a real-time data processing pipeline for social media sentiment analysis using Apache Kafka and Spark.": "Data Engineer",
    "Implement a data lake solution using Delta Lake and Apache Spark for large-scale log analytics.": "Data Engineer",
    "Develop a comprehensive environmental impact assessment for a new solar farm installation": "Environmental Consultant",
    "Design a physics-based puzzle game with realistic object interactions": "Game Developer",
    "Create a user-centered design system for a healthcare mobile app focusing on elderly users": "UI/UX Designer",
    "Design an interactive virtual laboratory system for high school physics experiments": "Education Specialist",
    "Create a cognitive behavioral therapy-based anxiety management program for remote workers": "Psychology Counselor",
    "Develop a career advancement strategy for a data analyst moving into management": "Career Counselor",
    "Draft a intellectual property protection strategy to protect my company": "Legal Advisor",
    "Create an influencer marketing campaign strategy for a sustainable fashion brand": "Marketing Strategist",
    "Develop a retirement planning strategy for a couple in their 40s with focus on ESG investments": "Financial Advisor",
    "Design a hybrid cloud architecture for a healthcare provider with strict data residency requirements": "Cloud Architect",
    "Design a waste management and recycling optimization plan for a large manufacturing facility": "Environmental Consultant",
    "Create a content strategy for launching a food and culture travel YouTube channel": "Traveling Blogger",
    "Design a diet plan for a vegetarian marathon runner": "Nutrition Consultant",
    "Create a series of posts and social media content documenting a culinary tour through Southeast Asia": "Traveling Blogger",
    "Design an end-to-end test automation framework for a microservices-based e-commerce platform using Selenium and TestNG": "QA Automation Engineer",
    "Create a performance testing strategy for a high-traffic mobile application using JMeter and Gatling": "QA Automation Engineer",
    "Create a Django-based laboratory management system with Pandas for equipment tracking, pytest fixtures for test data generation, and Celery for automated maintenance scheduling": "Python Developer",
}


def get_benchmark_metrics(predictions: dict) -> dict:
    """
    Calculate benchmark metrics from predictions

    Args:
        predictions: Dict mapping queries to predicted agents

    Returns:
        Dict containing accuracy metrics
    """

    total = len(QUERY_AGENT_MAPPING)
    correct = sum(
        1
        for query, agent in predictions.items()
        if agent == QUERY_AGENT_MAPPING.get(query)
    )

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

    results = []

    for query, predicted_agent in predictions.items():
        correct_agent = QUERY_AGENT_MAPPING.get(query)
        results.append(
            {
                "query": query,
                "predicted_agent": predicted_agent,
                "correct_agent": correct_agent,
                "is_correct": predicted_agent == correct_agent,
            }
        )

    return results
