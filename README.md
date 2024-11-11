<div align="center">
<h1>Agent Selection Algorithm Implementation</h1>
<h3>Universa AI Challenge Report</h3>
</div>

## Table of Contents
- [Project Overview](#project-overview)
- [Development Journey](#development-journey)
  - [Initial Phase](#initial-phase)
  - [Algorithm Evolution](#algorithm-evolution)
  - [Model Selection](#model-selection)
  - [Optimization Phase](#optimization-phase)
- [Technical Implementation](#technical-implementation)
- [Results and Performance](#results-and-performance)
- [Challenges and Solutions](#challenges-and-solutions)
- [Team Reflections and Learning Journey](#team-reflections-and-learning-journey)

## Project Overview
This project focuses on developing an efficient agent selection algorithm for Universa's AI system. The primary goal was to create an accurate matching system between user queries and available AI agents, considering various factors such as semantic similarity, agent ratings, and amount of rated responses.

## Development Journey

### Initial Phase
- **Understanding the Framework**
  - Familiarized with vector-based databases and ChromaDB
  - Studied the provided case study and repository structure
  - Analyzed the requirements and initial dataset

- **First Implementation**
  - Developed a basic algorithm using ChromaDB's default embedding function
  - Implemented basic rating weight system
  - Identified limitations in accuracy with default embeddings

### Algorithm Evolution
- **After First Mentor Session**
  - Recognized the need for superior embedding models
  - Shifted focus to testing various embedding approaches
  - Explored integration with AutoModel and Sentence Transformer

### Model Selection
- **Embedding Model Analysis**
  - Tested multiple embedding models for accuracy and performance
  - Selected Stella model (1.5B parameters) as optimal solution
  - Key factors in selection:
    - Superior accuracy in query matching
    - Reasonable initialization time
    - Balanced performance metrics

### Optimization Phase
- **System Improvements**
  - Implemented document caching for improved performance
  - Expanded agent database and query test sets
  - Developed custom benchmarking system
  - **Final Pipeline Optimization**
    - Cleaned up redundant code paths
    - Consolidated scattered implementations
    - Streamlined the main selection algorithm
    - Improved code readability and maintainability

- **Algorithm Refinements**
  - Incorporated weighted average rating system based on rated response volume
  - Added lexical distance component
  - Example: Rating weight calculation
    ```
    8/10 rating with 300 responses > 10/10 rating with 5 
    ```

## Technical Implementation
- **Core Components**
  - Stella 1.5B parameter model for embeddings
  - Caching system for agent documents
  - Hybrid similarity scoring:
    - Semantic similarity (primary weight)
    - Response-weighted rating system
    - Lexical distance measurement

## Results and Performance
- Improved accuracy in agent matching
- Efficient query processing through caching
- Balanced consideration of:
  - Semantic relevance
  - Performance (ratings)
  - Rated response volume
  - Lexical similarity

## Challenges and Solutions
1. **Initial Accuracy Issues**
   - Challenge: Poor performance with default embeddings
   - Solution: Implemented custom embedding model (Stella 1.5B)

2. **Performance Optimization**
   - Challenge: Slow document loading
   - Solution: Implemented caching system

3. **Rating System Reliability**
   - Challenge: Misleading ratings with low response counts
   - Solution: Developed response-weighted rating system

4. **Cost Consideration**
   - Challenge: Initially considered input/output costs
   - Solution: Refocused on accuracy after mentor session

## Team Reflections and Learning Journey

### Initial Challenges
- **Repository Complexity**
  - Initially overwhelmed by the extensive codebase
  - Faced steep learning curve with unfamiliar technologies
  - Needed time to understand the interconnections between different components

### Growth and Adaptation
- **Progressive Understanding**
  - Gradually developed better grasp of the system architecture
  - Learned to navigate and modify the codebase effectively
  - Built confidence in implementing new features

### Mentor Impact
- **Guidance and Direction**
  - Mentor sessions provided crucial clarity on project priorities
  - Helped validate or redirect our technical approaches
  - Offered valuable insights for algorithm refinement

### Key Learnings
- **Technical Skills**
  - Gained practical experience with vector databases
  - Developed understanding of embedding models and their applications
  - Learned to balance multiple factors in algorithm design

- **Project Management**
  - Importance of systematic testing and benchmarking
  - Value of iterative development and continuous refinement
  - Benefits of seeking guidance when facing technical challenges

### Final Thoughts
Looking back, this challenge has been quite a journey for our team. What started as an intimidating project with unfamiliar concepts and a complex codebase turned into a very valuable learning experience. The most satisfying part was seeing our steady progress - each small victory, from getting our first accurate matches to implementing the final optimizations, built up our confidence and understanding. The final days were spent cleaning up our initially messy implementations, which really showed us how much we'd learned - we could now see clearly how to structure the code properly and remove redundancies that had crept in during our learning phase. This challenge taught us that it's okay to feel overwhelmed at first, what matters is taking it step by step and not being afraid to ask for help when needed.