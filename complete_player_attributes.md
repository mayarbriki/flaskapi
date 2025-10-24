# FIFA Players Dataset - Complete Attributes Visualization 

## Player Data Model

```mermaid
erDiagram
    PLAYER {
        string name
        string full_name
        date birth_date
        int age
        float height_cm
        float weight_kgs
        string positions
        string nationality
        int overall_rating
        int potential
        float value_euro
        float wage_euro
        string preferred_foot
        int international_reputation
        int weak_foot
        int skill_moves
        string body_type
        float release_clause_euro
    }

    NATIONAL_TEAM_INFO {
        string national_team
        int national_rating
        string national_team_position
        int national_jersey_number
    }

    TECHNICAL_SKILLS {
        int crossing
        int finishing
        int heading_accuracy
        int short_passing
        int volleys
        int dribbling
        int curve
        int freekick_accuracy
        int long_passing
        int ball_control
    }

    PHYSICAL_ATTRIBUTES {
        int acceleration
        int sprint_speed
        int agility
        int reactions
        int balance
        int shot_power
        int jumping
        int stamina
        int strength
    }

    MENTAL_ATTRIBUTES {
        int long_shots
        int aggression
        int interceptions
        int positioning
        int vision
        int penalties
        int composure
    }

    DEFENSIVE_SKILLS {
        int marking
        int standing_tackle
        int sliding_tackle
    }

    PLAYER ||--|| NATIONAL_TEAM_INFO : has
    PLAYER ||--|| TECHNICAL_SKILLS : possesses
    PLAYER ||--|| PHYSICAL_ATTRIBUTES : has
    PLAYER ||--|| MENTAL_ATTRIBUTES : possesses
    PLAYER ||--|| DEFENSIVE_SKILLS : has
```

## Attribute Categories Breakdown

```mermaid
mindmap
    root((Player Attributes))
        Personal Info
            Name & Full Name
            Birth Date & Age
            Height & Weight
            Nationality
            Body Type
        Contract Details
            Value
            Wage
            Release Clause
        Basic Stats
            Overall Rating
            Potential
            Positions
            Preferred Foot
        International
            National Team
            National Rating
            Team Position
            Jersey Number
        Technical
            Crossing
            Finishing
            Heading
            Passing
            Volleys
            Dribbling
            Curve
            Free Kicks
            Ball Control
        Physical
            Acceleration
            Sprint Speed
            Agility
            Reactions
            Balance
            Shot Power
            Jumping
            Stamina
            Strength
        Mental
            Long Shots
            Aggression
            Interceptions
            Positioning
            Vision
            Penalties
            Composure
        Defensive
            Marking
            Standing Tackle
            Sliding Tackle
```

## Data Flow and Processing

```mermaid
graph TB
    subgraph Input_Data
        A[Raw Player Data CSV]
        B[Player Statistics]
        C[Team Information]
    end

    subgraph Processing_Layer
        D[Data Cleaning]
        E[Attribute Normalization]
        F[Feature Engineering]
        G[Statistical Analysis]
    end

    subgraph Player_Analysis
        H[Technical Assessment]
        I[Physical Evaluation]
        J[Mental Analysis]
        K[Defensive Analysis]
    end

    subgraph Output_Features
        L[Player Recommendations]
        M[Performance Stats]
        N[Comparison Tools]
        O[Scouting Reports]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    G --> I
    G --> J
    G --> K
    H --> L
    I --> L
    J --> L
    K --> L
    H --> M
    I --> M
    J --> M
    K --> M
    M --> N
    L --> O

    style A fill:#2563eb,color:#fff
    style B fill:#2563eb,color:#fff
    style C fill:#2563eb,color:#fff
    style D fill:#64748b,color:#fff
    style E fill:#64748b,color:#fff
    style F fill:#64748b,color:#fff
    style G fill:#64748b,color:#fff
    style H fill:#059669,color:#fff
    style I fill:#059669,color:#fff
    style J fill:#059669,color:#fff
    style K fill:#059669,color:#fff
    style L fill:#e11d48,color:#fff
    style M fill:#e11d48,color:#fff
    style N fill:#e11d48,color:#fff
    style O fill:#e11d48,color:#fff
```

## Skill Ratings Scale

```mermaid
graph LR
    subgraph Rating_Scale
        A[0-20: Very Poor] --> B[21-40: Poor]
        B --> C[41-60: Average]
        C --> D[61-80: Good]
        D --> E[81-90: Very Good]
        E --> F[91-99: Excellent]
    end

    subgraph Special_Ratings
        G[International Reputation: 1-5]
        H[Weak Foot: 1-5]
        I[Skill Moves: 1-5]
    end

    style A fill:#ef4444,color:#fff
    style B fill:#f97316,color:#fff
    style C fill:#eab308,color:#fff
    style D fill:#84cc16,color:#fff
    style E fill:#22c55e,color:#fff
    style F fill:#2563eb,color:#fff
    style G fill:#8b5cf6,color:#fff
    style H fill:#8b5cf6,color:#fff
    style I fill:#8b5cf6,color:#fff
```

## Value and Contract Structure

```mermaid
graph TD
    subgraph Player_Value
        A[Market Value] --> B[Base Value]
        A --> C[Potential Value]
        A --> D[Release Clause]
    end

    subgraph Contract_Details
        E[Wages] --> F[Weekly Wage]
        E --> G[Annual Salary]
    end

    subgraph Value_Factors
        H[Age]
        I[Overall Rating]
        J[Potential]
        K[International Rep]
    end

    H --> A
    I --> A
    J --> A
    K --> A

    style A fill:#2563eb,color:#fff
    style B fill:#64748b,color:#fff
    style C fill:#64748b,color:#fff
    style D fill:#64748b,color:#fff
    style E fill:#059669,color:#fff
    style F fill:#e11d48,color:#fff
    style G fill:#e11d48,color:#fff
    style H fill:#8b5cf6,color:#fff
    style I fill:#8b5cf6,color:#fff
    style J fill:#8b5cf6,color:#fff
    style K fill:#8b5cf6,color:#fff
```

## Player Performance Metrics

```mermaid
graph TD
    subgraph Technical_Ability
        A[Ball Skills] --> A1[Dribbling]
        A[Ball Skills] --> A2[Ball Control]
        A[Ball Skills] --> A3[First Touch]
        
        B[Shooting] --> B1[Finishing]
        B[Shooting] --> B2[Shot Power]
        B[Shooting] --> B3[Long Shots]
        
        C[Passing] --> C1[Short Passing]
        C[Passing] --> C2[Long Passing]
        C[Passing] --> C3[Vision]
    end

    subgraph Physical_Capability
        D[Speed] --> D1[Acceleration]
        D[Speed] --> D2[Sprint Speed]
        
        E[Strength] --> E1[Physical Power]
        E[Strength] --> E2[Jumping]
        E[Strength] --> E3[Stamina]
    end

    subgraph Mental_Attributes
        F[Game Intelligence] --> F1[Positioning]
        F[Game Intelligence] --> F2[Interceptions]
        F[Game Intelligence] --> F3[Composure]
    end

    style A fill:#2563eb,color:#fff
    style B fill:#2563eb,color:#fff
    style C fill:#2563eb,color:#fff
    style D fill:#059669,color:#fff
    style E fill:#059669,color:#fff
    style F fill:#e11d48,color:#fff
```

## Player Attributes Details

### Personal Information
- Name and Full Name
- Birth Date and Age
- Height (cm) and Weight (kg)
- Nationality
- Body Type
- Positions

### Performance Ratings
- Overall Rating (0-99)
- Potential Rating (0-99)
- International Reputation (1-5)
- Weak Foot Rating (1-5)
- Skill Moves Rating (1-5)

### Technical Attributes
- Ball Control and Dribbling
- Passing (Short and Long)
- Shooting and Finishing
- Free Kicks and Penalties
- Crossing and Curve

### Physical Attributes
- Speed (Acceleration and Sprint)
- Strength and Stamina
- Agility and Balance
- Jumping
- Reactions

### Mental Attributes
- Vision and Composure
- Positioning
- Aggression
- Interceptions
- Long Shots

### Defensive Attributes
- Marking
- Standing Tackle
- Sliding Tackle

### Economic Values
- Market Value (Euro)
- Wage (Euro)
- Release Clause (Euro)