Soccer Player Analytics Tool with AI Integration
Project Overview

This mini-project, developed as part of the "IA for Software Engineering" course (5SAE), is a web-based application for analyzing soccer player data.
It leverages AI to provide intelligent insights such as player performance predictions, anomaly detection in statistics, and personalized recommendations for team compositions.

The tool integrates an external AI API (e.g., OpenAI's GPT for natural language queries on player data) and uses AI-assisted code generation for features like automated testing and refactoring suggestions.

The application processes a CSV dataset of player attributes (e.g., ratings, physical stats, skills) to generate visualizations, detect outliers (e.g., unusual stat combinations), and suggest improvements.
Development incorporated generative AI tools (ChatGPT, GitHub Copilot) for prompt-based code generation, test creation, and architecture design.


Technologies Used

Backend: Python (Flask)

AI Integration: OpenAI API for query processing and anomaly detection

Functional Requirements

The system must provide the following core functionalities:

1. Data Ingestion and Management

Upload and parse CSV files containing player data (e.g., name, age, overall_rating, skills).

Store player records in a database with CRUD operations (Create, Read, Update, Delete).

Support querying players by filters (e.g., nationality, position, rating > 80).

2. Player Analytics and Visualization

Generate summary statistics (e.g., average age by nationality, top skills distribution).

Display interactive charts for player comparisons (e.g., radar charts for dribbling, passing).

Export analytical reports as PDF or CSV files.

3. AI-Powered Insights

Integrate an AI API to process natural language queries (e.g.,
"Recommend a striker for Argentina under 30 years old"), returning ranked player suggestions.

Detect anomalies in player data (e.g., unusually high skill_moves but low stamina) using statistical or ML models.

Provide AI-generated refactoring suggestions for code efficiency and structure improvement.

4. Automated Code and Testing Generation

Use generative AI to automatically create unit tests for backend endpoints (target coverage > 80%).

Generate boilerplate code for new features (e.g., adding a new skill metric).

5. Refactoring and Optimization Recommendations

Perform AI-driven code scans to detect code smells (e.g., duplicated logic).

Suggest optimizations (e.g., “Extract method for skill calculation”).

Non-Functional Requirements

The system must meet specific quality attributes to ensure performance, usability, and maintainability.

1. Performance

Response Time: < 2 seconds for queries on up to 1,000 player records.

Throughput: Support at least 50 concurrent users without degradation.

AI API Calls: Response < 5 seconds per inference, with fallback to cached results.

2. Usability

Interface: Responsive UI for both desktop and mobile devices.

Accessibility: WCAG 2.1 AA compliance (e.g., alt text for charts, keyboard navigation).

Error Handling: Clear, descriptive messages (e.g., “Invalid CSV format: Missing ‘overall_rating’ column”).


4. Reliability and Maintainability

Uptime: 99% availability, with automated database backups.

Code Quality: SonarQube score > 80%; AI-generated tests must pass CI/CD validation.

Scalability: Modular architecture to support additional datasets (e.g., teams, leagues).

5. Compatibility

Data Formats: CSV (UTF-8), JSON for API responses.

Implemented APIs

The application exposes a RESTful API backend built with Flask to handle data operations, analytics, and AI integrations.
The API is stateless, uses JSON payloads, and follows standard HTTP status codes.
Authentication is enforced through JWT tokens for all protected endpoints.

All endpoints are prefixed with /api/v1/.

1. Data Ingestion and Management

Players can be queried using filters such as nationality, position, or overall rating (e.g., rating > 80).

2. Player Analytics and Visualization

Generate summary statistics (e.g., average age by nationality, top 10 players by rating).

Provide interactive visualizations (e.g., radar charts for skill comparisons).

3. AI-Powered Insights

Handle natural language queries via AI API (e.g.,
“Recommend a striker for Argentina under 30 years old”).

Detect anomalies in player data using ML or statistical analysis.

Offer AI-based refactoring and project optimization suggestions.

Non-Functional Requirements (Detailed)
Performance

Response time: < 2 seconds for queries on up to 1,000 player records.

Handle at least 50 concurrent users without performance degradation.

AI inference: ≤ 5 seconds with caching fallback.

Usability

Responsive design for all devices.

Accessibility compliance: WCAG 2.1 AA (alt text, keyboard navigation, high contrast).

Friendly error feedback (e.g., “Invalid CSV format: Missing overall_rating column”).

Reliability and Maintainability

99% uptime with automatic backups.

Modular structure to accommodate new datasets (teams, leagues, etc.).
