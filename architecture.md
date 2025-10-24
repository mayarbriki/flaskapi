# FIFA Player Recommender - Architecture Overview

```mermaid
graph TD
    %% Main Components
    Client[Browser Client]
    Flask[Flask Server]
    DB[(FIFA Players DB)]
    ML[ML Model]

    %% Static and Templates
    Static[Static Files]
    Templates[HTML Templates]

    %% Static File Details
    CSS[styles.css]
    JS[app.js]

    %% Template Details
    IndexHTML[index.html]
    AuthHTML[auth.html]

    %% Main Features
    Search[Player Search]
    Recommend[Recommendations]
    Stats[Statistics]
    Browse[Browse Players]
    Auth[Authentication]

    %% Connections - Client Side
    Client --> |HTTP Requests| Flask
    Flask --> |JSON Responses| Client

    %% Static and Template Connections
    Flask --> Static
    Flask --> Templates
    Static --> CSS
    Static --> JS
    Templates --> IndexHTML
    Templates --> AuthHTML

    %% Data Flow
    Flask --> DB
    Flask --> ML
    ML --> |Predictions| Flask
    DB --> |Player Data| Flask

    %% Features
    Flask --> Search
    Flask --> Recommend
    Flask --> Stats
    Flask --> Browse
    Flask --> Auth

    %% Styling
    classDef primary fill:#2563eb,stroke:#fff,stroke-width:2px,color:#fff
    classDef secondary fill:#64748b,stroke:#fff,stroke-width:2px,color:#fff
    classDef highlight fill:#059669,stroke:#fff,stroke-width:2px,color:#fff

    class Flask,ML primary
    class DB,Static,Templates secondary
    class Search,Recommend,Stats,Browse,Auth highlight

```

## Component Description

### Core Components
- **Flask Server**: Main backend server handling all requests and business logic
- **ML Model**: Machine learning model for player recommendations and predictions
- **FIFA Players DB**: Database storing all player information and statistics

### Frontend
- **Static Files**:
  - `styles.css`: Global styling and UI components
  - `app.js`: Client-side JavaScript for interactivity
- **Templates**:
  - `index.html`: Main application interface
  - `auth.html`: Authentication pages

### Main Features
1. **Player Search**: Advanced search with multiple filters
2. **Recommendations**: AI-powered player recommendations
3. **Statistics**: Global stats and player rankings
4. **Browse Players**: Interactive player browsing interface
5. **Authentication**: User management and security

### Data Flow
- Client makes requests to Flask server
- Server interacts with database and ML model
- Responses are sent back as JSON
- Frontend renders data using JavaScript
- Real-time updates and interactive UI

## Technology Stack
- **Backend**: Python/Flask
- **Frontend**: HTML, CSS, JavaScript
- **Database**: CSV/DataFrame processing
- **ML**: Scikit-learn for recommendations
- **UI**: Custom responsive design