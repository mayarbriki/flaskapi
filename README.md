Soccer Player Analytics Tool with AI Integration
Project Overview
This mini-project, developed as part of the "IA for Software Engineering" course (5SAE), is a web-based application for analyzing soccer player data. It leverages AI to provide intelligent insights, such as player performance predictions, anomaly detection in statistics, and personalized recommendations for team compositions. The tool integrates an external AI API (e.g., OpenAI's GPT for natural language queries on player data) and uses AI-assisted code generation for features like automated testing and refactoring suggestions.
The application processes a CSV dataset of player attributes (e.g., ratings, physical stats, skills) to generate visualizations, detect outliers (e.g., unusual stat combinations), and suggest improvements. Development incorporated generative AI tools (ChatGPT, GitHub Copilot) for prompt-based code generation, test creation, and architecture design.
Technologies Used:

Backend: Python (Flask)
AI Integration: OpenAI API for query processing and anomaly detection
Testing: Pytest with AI-generated test cases

Update Date: October 22, 2025 – Incorporated detailed API documentation based on implemented endpoints.
Functional Requirements
The system must provide the following core functionalities:

Data Ingestion and Management

Upload and parse CSV files containing player data (e.g., attributes like name, age, overall_rating, skills).
Store player records in a database with CRUD operations (Create, Read, Update, Delete).
Support querying players by filters (e.g., nationality, position, rating > 80).


Player Analytics and Visualization

Generate summary statistics (e.g., average age by nationality, top skills distribution).
Display interactive charts for player comparisons (e.g., radar charts for skill sets like dribbling, passing).
Export reports as PDF/CSV.


AI-Powered Insights

Integrate an AI API to process natural language queries (e.g., "Recommend a striker for Argentina under 30 years old") and return ranked player suggestions based on weighted attributes.
Detect anomalies in player data (e.g., flag players with unusually high skill_moves but low stamina using statistical models or ML thresholds).
Provide refactoring recommendations for code modules (meta-feature: AI scans project code for inefficiencies and suggests optimizations).


Automated Code and Testing Generation

Use generative AI (via prompts) to auto-generate unit tests for backend endpoints (e.g., coverage > 80% for data ingestion).
Generate boilerplate code for new features (e.g., adding a new skill metric).



Refactoring and Optimization Recommendations

AI-driven scan of the codebase to detect code smells (e.g., duplicated logic in analytics functions) and suggest refactors (e.g., "Extract method for skill calculation").



Non-Functional Requirements
The system must adhere to the following quality attributes to ensure reliability, usability, and scalability:

Performance

Response time: < 2 seconds for queries on up to 1,000 player records.
Throughput: Handle 50 concurrent users without degradation.
AI API calls: Limit to < 5 seconds per inference, with fallback to cached results.


Usability

Intuitive UI: Responsive design for desktop/mobile, with tooltips for attributes (e.g., "weak_foot: 1-5 rating").
Accessibility: WCAG 2.1 AA compliance (e.g., alt text for charts, keyboard navigation).
Error Handling: User-friendly messages (e.g., "Invalid CSV format: Missing 'overall_rating' column").



Compatibility

Data Formats: CSV (UTF-8), JSON for API responses.
Implemented APIs
The application exposes a RESTful API backend using Flask to handle data operations, analytics, and AI integrations. The API is stateless, uses JSON payloads, and follows standard HTTP status codes. Authentication via JWT tokens is required for protected endpoints.
All endpoints are prefixed with /api/v1/. Below is a comprehensive list, grouped by functionality.
Implemented APIs
The application exposes a RESTful API backend using Flask to handle data operations, analytics, and AI integrations. The API is stateless, uses JSON payloads, and follows standard HTTP status codes. Authentication via JWT tokens is required for protected endpoints.
All endpoints are prefixed with /api/v1/. Below is a comprehensive list, grouped by functionality.
- Functional Requirements
Data Ingestion and Management

Players can be queried by filters, such as nationality, position, or overall rating (e.g., rating > 80).

 Player Analytics and Visualization

Generate summary statistics (e.g., average age by nationality, top 10 players by overall rating).

Provide interactive visualizations, such as radar charts for player skill comparisons.


AI-Powered Insights

Integrate an AI API (e.g., OpenAI) to handle natural language queries, such as:

“Recommend a striker for Argentina under 30 years old.”

Detect anomalies in player data, such as inconsistent skill ratings, using machine learning or statistical thresholds.

Provide AI-driven recommendations for code refactoring and project optimization.

 Non-Functional Requirements
 Performance

Response time: < 2 seconds for queries on up to 1,000 player records.

Throughput: Handle at least 50 concurrent users without performance degradation.

AI inference: API responses within 5 seconds, with fallback to cached results when needed.

Usability

Responsive UI for desktop and mobile devices.

Accessibility compliance: WCAG 2.1 AA (alt text, keyboard navigation, high contrast).

Error handling: Clear and friendly messages (e.g., “Invalid CSV format: Missing overall_rating column”).


 Reliability and Maintainability

Uptime: 99% system availability with automatic database backups.

Code quality: Maintain a SonarQube score > 80%; all AI-generated tests must pass CI/CD checks.

Maintainability: Modular codebase to support adding new datasets (e.g., teams, leagues).

