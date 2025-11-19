# AI-Assisted Student Onboarding System

## Semester 1 Timeline

```mermaid
gantt
    title Semester 1 - Dissertation Progress
    dateFormat  YYYY-MM-DD
    axisFormat  %b %d

    section Literature Review
    Initial reading & research    :lit1, 2025-09-15, 2w
    Identify key papers           :lit2, after lit1, 1w
    Write literature review       :lit3, after lit2, 3w

    section Requirements & Design
    Requirements gathering        :req1, 2025-10-06, 2w
    System design                 :des1, after req1, 2w
    Design documentation          :des2, after des1, 1w

    section Implementation
    Environment setup             :imp1, 2025-10-27, 1w
    Core functionality            :imp2, after imp1, 3w
    Initial testing               :imp3, after imp2, 1w

    section Documentation
    Progress report               :doc1, 2025-11-24, 2w
    Interim submission            :milestone, crit, 2025-12-08, 1d
```

## Semester 2 Timeline (12 January - 26 March 2026)

Implementation timeline for the AI-powered onboarding application for Heriot-Watt University new and international students.

```mermaid
gantt
    title AI-Assisted Student Onboarding System - Implementation Timeline
    dateFormat YYYY-MM-DD

    section Week 1 (12-18 Jan)
    Environment Setup & Data Collection             :w1, 2026-01-12, 7d

    section Week 2 (19-25 Jan)
    Document Processing & Chunking                  :w2, 2026-01-19, 7d

    section Week 3 (26 Jan-1 Feb)
    Vector Database Setup & Embeddings              :w3, 2026-01-26, 7d

    section Week 4 (2-8 Feb)
    RAG Pipeline & Azure OpenAI Integration         :w4, 2026-02-02, 7d

    section Week 5 (9-15 Feb)
    Multilingual Support & Language Detection       :w5, 2026-02-09, 7d

    section Week 6 (16-22 Feb)
    Backend API Development                         :w6, 2026-02-16, 7d

    section Week 7 (23 Feb-1 Mar)
    Frontend Development & UI Implementation        :w7, 2026-02-23, 7d

    section Week 8 (2-8 Mar)
    System Integration & Testing                    :w8, 2026-03-02, 7d

    section Week 9 (9-15 Mar)
    User Evaluation & Testing Sessions              :crit, w9, 2026-03-09, 7d

    section Week 10 (16-22 Mar)
    Results Analysis & Implementation Write-up      :w10, 2026-03-16, 7d

    section Week 11 (23-26 Mar)
    Final Proofreading & Submission                 :crit, w11, 2026-03-23, 4d
    Final Submission                                :milestone, submit, 2026-03-26, 0d
```

## Key Milestones

- **Week 9 (9-15 Mar)**: User Testing Sessions (Critical)
- **Week 11 (26 Mar)**: Final Submission Deadline

## Technologies

- **Backend**: Flask/FastAPI, Azure OpenAI Service
- **Vector Database**: FAISS/ChromaDB
- **Frontend**: React/Vue.js
- **Languages**: English, Mandarin, Hindi, Arabic
