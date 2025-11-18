# AI-Assisted Student Onboarding System

## Project Timeline (January 12 - March 26, 2025)

Implementation timeline for the AI-powered onboarding application for Heriot-Watt University new and international students.

```mermaid
gantt
    title AI-Assisted Student Onboarding System - Implementation Timeline
    dateFormat YYYY-MM-DD

    section Phase 1: Data & Setup
    Environment Setup & Tools Installation           :done, setup, 2025-01-12, 3d
    Data Collection (HWU Sources)                   :done, data1, 2025-01-12, 7d
    Document Normalization & Cleaning               :active, data2, 2025-01-15, 7d
    Chunking & Metadata Tagging                     :data3, 2025-01-19, 5d

    section Phase 2: RAG Pipeline
    Embedding Model Selection & Testing             :rag1, 2025-01-22, 4d
    Vector Database Setup (FAISS/ChromaDB)          :rag2, 2025-01-26, 5d
    Query Processing & Retrieval Logic              :rag3, 2025-01-29, 6d
    Azure OpenAI Integration                        :rag4, 2025-02-02, 5d
    Prompt Engineering & Response Generation        :rag5, 2025-02-05, 5d

    section Phase 3: Multilingual Support
    Language Detection Implementation               :lang1, 2025-02-08, 4d
    Translation Integration                         :lang2, 2025-02-10, 5d
    Cross-Language Retrieval Testing                :lang3, 2025-02-12, 4d

    section Phase 4: Backend & API
    Backend API Development (Flask/FastAPI)         :back1, 2025-02-10, 7d
    Post-Processing Pipeline                        :back2, 2025-02-15, 5d
    Source Attribution & Citation Formatting        :back3, 2025-02-17, 4d
    API Security & Rate Limiting                    :back4, 2025-02-19, 3d

    section Phase 5: Frontend
    UI Design & Wireframing                         :front1, 2025-02-16, 4d
    Frontend Development (React/Vue.js)             :front2, 2025-02-20, 7d
    Mobile Responsive Design                        :front3, 2025-02-24, 4d
    Accessibility Implementation (WCAG 2.1)         :front4, 2025-02-26, 4d

    section Phase 6: Integration & Testing
    System Integration                              :int1, 2025-02-27, 5d
    Performance Testing & Optimization              :int2, 2025-03-01, 4d
    Accuracy Testing (95% threshold)                :int3, 2025-03-03, 4d
    Bug Fixes & Refinement                          :int4, 2025-03-05, 4d

    section Phase 7: User Evaluation
    Participant Recruitment                         :eval1, 2025-02-24, 7d
    User Testing Sessions                           :crit, eval2, 2025-03-09, 5d
    SUS Scoring & Quantitative Analysis             :eval3, 2025-03-12, 3d
    Qualitative Interviews & Thematic Analysis      :eval4, 2025-03-13, 4d

    section Phase 8: Final Deliverables
    Results Analysis & Interpretation               :final1, 2025-03-15, 4d
    Implementation Section Write-up                 :final2, 2025-03-17, 4d
    Evaluation Section Write-up                     :final3, 2025-03-19, 4d
    Final Proofreading & Formatting                 :final4, 2025-03-22, 3d
    Submission Preparation                          :crit, final5, 2025-03-25, 2d
    Final Submission                                :milestone, submit, 2025-03-26, 0d
```

## Key Milestones

- **March 9-13**: User Testing Sessions (Critical)
- **March 26**: Final Submission Deadline

## Technologies

- **Backend**: Flask/FastAPI, Azure OpenAI Service
- **Vector Database**: FAISS/ChromaDB
- **Frontend**: React/Vue.js
- **Languages**: English, Mandarin, Hindi, Arabic
