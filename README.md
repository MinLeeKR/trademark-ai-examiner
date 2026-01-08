# 🏛️ AI 상표 심사관 (Trademark AI Examiner)

> LLM 기반 차세대 상표 선행 조사 및 침해 분석 시스템 아키텍처

한국 상표법의 **외관(Appearance)**, **호칭(Pronunciation)**, **관념(Conception)** 3요소를 AI로 분석하여 상표 유사성을 판단하는 시스템입니다.

---

## 📋 목차

1. [실제 분석 예시: "라라스윗"](#-실제-분석-예시-라라스윗)
2. [시스템 아키텍처 (개념 중심)](#-시스템-아키텍처-개념-중심)
3. [시스템 아키텍처 (알고리즘 중심)](#-시스템-아키텍처-알고리즘-중심)
4. [핵심 기술](#-핵심-기술)
5. [문서](#-문서)

---

## 🎯 실제 분석 예시: "라라스윗"

**"라라스윗(Lalasweet)"** 상표를 출원한다고 가정했을 때, AI 시스템이 수행하는 분석 흐름입니다.

### 출원 상표

![라라스윗 로고](assets/lalasweet_logo.png)

### AI 분석 프로세스

```mermaid
flowchart LR
    subgraph 입력정보["📥 출원 상표 정보"]
        상표명["상표명: 라라스윗<br/>영문: Lalasweet"]
        로고["로고: 파란 배경<br/>흰색 필기체<br/>L자 곡선 장식"]
        지정상품["지정상품: 제30류<br/>과자, 캔디, 초콜릿<br/>아이스크림"]
    end

    subgraph 호칭분석["🔊 호칭 유사성 검사"]
        발음["발음 분석<br/>━━━━━━━━━<br/>[라라스윗]<br/>4음절, 'ㄹ' 반복"]
        호칭후보["유사 호칭 후보<br/>━━━━━━━━━<br/>· 라라스위트<br/>· 랄라스윗<br/>· 라라슈윗<br/>· 스윗라라"]
        호칭결과["⚠️ 호칭 유사도<br/>━━━━━━━━━<br/>라라스위트: 95%<br/>랄라스윗: 82%<br/>스윗라라: 71%"]
    end

    subgraph 외관분석["👁️ 외관 유사성 검사"]
        로고특징["로고 특징 추출<br/>━━━━━━━━━<br/>· 파란색 배경<br/>· 흰색 필기체<br/>· L자 곡선 디자인<br/>· 라운드 폰트"]
        외관후보["유사 외관 후보<br/>━━━━━━━━━<br/>· 유사 필기체 로고<br/>· 파란+흰 조합 로고<br/>· 곡선 L 디자인"]
        외관결과["✅ 외관 판정<br/>━━━━━━━━━<br/>동일/유사 로고<br/>발견되지 않음"]
    end

    subgraph 관념분석["💡 관념 유사성 검사"]
        의미분석["의미 분석<br/>━━━━━━━━━<br/>라라 = 즐거움, 노래<br/>스윗 = 달콤함<br/>→ '즐거운 달콤함'"]
        관념후보["유사 관념 후보<br/>━━━━━━━━━<br/>· 스위트드림<br/>· 해피스윗<br/>· 달콤한하루<br/>· 스윗멜로디"]
        관념결과["⚠️ 관념 유사도<br/>━━━━━━━━━<br/>해피스윗: 68%<br/>스위트드림: 55%<br/>직접 충돌 낮음"]
    end

    subgraph 종합판단["⚖️ AI 심사관 종합 판단"]
        판례검토["관련 판례 검토<br/>━━━━━━━━━<br/>·'스윗' 단독은 식별력 약함<br/>·'라라+X' 결합상표 선례<br/>·제30류 유사 판단 기준"]
        결론["📋 최종 의견<br/>━━━━━━━━━<br/>등록 가능성: 중간<br/><br/>⚠️ 주의사항:<br/>·'라라스위트' 선등록 확인 필요<br/>·제30류 내 유사상표 존재<br/><br/>💡 권고사항:<br/>·출원 전 의견서 준비<br/>·지정상품 범위 조정 검토"]
    end

    %% 흐름 연결
    상표명 --> 발음
    로고 --> 로고특징
    상표명 --> 의미분석
    지정상품 --> 판례검토

    발음 --> 호칭후보
    호칭후보 --> 호칭결과
    
    로고특징 --> 외관후보
    외관후보 --> 외관결과
    
    의미분석 --> 관념후보
    관념후보 --> 관념결과

    호칭결과 --> 판례검토
    외관결과 --> 판례검토
    관념결과 --> 판례검토
    판례검토 --> 결론

    %% 스타일링
    classDef inputStyle fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef phoneticStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef visualStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef conceptStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef judgeStyle fill:#fff8e1,stroke:#ff8f00,stroke-width:2px
    classDef warningStyle fill:#ffebee,stroke:#c62828,stroke-width:2px
    classDef okStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px

    class 상표명,로고,지정상품 inputStyle
    class 발음,호칭후보 phoneticStyle
    class 로고특징,외관후보 visualStyle
    class 의미분석,관념후보 conceptStyle
    class 판례검토,결론 judgeStyle
    class 호칭결과,관념결과 warningStyle
    class 외관결과 okStyle
```

### 분석 결과 요약

| 분석 항목 | 결과 | 비고 |
|:----------|:----:|:-----|
| **호칭 유사성** | ⚠️ 주의 | "라라스위트" 95% 유사 |
| **외관 유사성** | ✅ 안전 | 유사 로고 미발견 |
| **관념 유사성** | ⚠️ 주의 | "해피스윗" 68% 유사 |
| **등록 가능성** | 🟡 중간 | 의견서 준비 권고 |

---

## 🏗️ 시스템 아키텍처 (개념 중심)

> 알고리즘 명칭 대신 **각 컴포넌트가 수행하는 역할**을 쉬운 키워드로 설명합니다.

```mermaid
flowchart TB
    subgraph 사용자["👤 사용자 인터페이스"]
        입력["상표 출원 정보 입력<br/>· 상표명, 로고 이미지<br/>· 지정 상품/서비스류"]
        결과["분석 결과 확인<br/>· 유사 상표 목록<br/>· 침해 가능성 판정<br/>· 근거 판례 인용"]
    end

    subgraph 두뇌["🧠 AI 심사관 (멀티 에이전트)"]
        분석관["1️⃣ 질의 분석<br/>━━━━━━━━━<br/>출원 상표 특징 파악<br/>검색 전략 수립<br/>키워드 도출"]
        조사관["2️⃣ 선행 상표 조사<br/>━━━━━━━━━<br/>호칭·외관·관념별<br/>유사 상표 검색<br/>관련 판례 수집"]
        심사관["3️⃣ 유사성 판단<br/>━━━━━━━━━<br/>법적 유사성 평가<br/>혼동 가능성 분석<br/>거절 이유 검토"]
        작성관["4️⃣ 보고서 작성<br/>━━━━━━━━━<br/>분석 결과 정리<br/>근거 판례 인용<br/>의견서 초안 생성"]
    end

    subgraph 추론["💭 법률 추론 엔진"]
        사고["단계별 사고<br/>━━━━━━━━━<br/>문제 → 가설 수립<br/>→ 정보 탐색<br/>→ 검증 → 결론"]
        생성["자연어 생성<br/>━━━━━━━━━<br/>법률 용어 사용<br/>논리적 문장 구성<br/>전문가 수준 답변"]
    end

    subgraph 검색["📚 지식 검색 시스템"]
        의미검색["의미 기반 검색<br/>━━━━━━━━━<br/>질문의 '의도' 파악<br/>의미적으로 유사한<br/>문서 찾기"]
        키워드검색["키워드 검색<br/>━━━━━━━━━<br/>정확한 용어 매칭<br/>사건번호, 조문 검색"]
        재정렬["검색 결과 정렬<br/>━━━━━━━━━<br/>법적 관련성 기준<br/>최적 순위 재배치"]
    end

    subgraph 호칭분석["🔊 발음 유사성 분석"]
        발음변환["발음으로 변환<br/>━━━━━━━━━<br/>글자 → 실제 소리<br/>예: '신라' → [실라]<br/>음운 변동 적용"]
        발음비교["발음 유사도 측정<br/>━━━━━━━━━<br/>자음·모음 패턴 비교<br/>청각적 혼동 가능성<br/>유사도 점수 산출"]
    end

    subgraph 외관분석["👁️ 시각적 유사성 분석"]
        특징추출["로고 특징 추출<br/>━━━━━━━━━<br/>형태, 색상, 패턴<br/>미세한 선/곡선 분석<br/>주파수 도메인 분석"]
        이미지비교["이미지 유사도 측정<br/>━━━━━━━━━<br/>전체 인상 비교<br/>부분 요소 매칭<br/>변형/모방 탐지"]
    end

    subgraph 관념분석["💡 의미적 유사성 분석"]
        의미추출["의미/인상 추출<br/>━━━━━━━━━<br/>로고가 주는 느낌<br/>브랜드 컨셉 파악<br/>텍스트-이미지 연결"]
        개념비교["관념 유사도 측정<br/>━━━━━━━━━<br/>다른 형태지만<br/>같은 의미인지 판단<br/>예: 🍎 vs 'Apple'"]
    end

    subgraph 데이터["📦 법률 데이터베이스"]
        상표DB["상표 데이터<br/>━━━━━━━━━<br/>KIPRIS 등록 상표<br/>출원 정보<br/>지정 상품류"]
        판례DB["판례 데이터<br/>━━━━━━━━━<br/>대법원 판결문<br/>유사 판단 기준<br/>선례 법리"]
    end

    %% 사용자 흐름
    입력 --> 분석관
    분석관 --> 조사관
    조사관 --> 심사관
    심사관 --> 작성관
    작성관 --> 결과

    %% 추론 연결
    두뇌 <--> 사고
    사고 <--> 생성

    %% 조사관의 도구 사용
    조사관 --> 의미검색
    조사관 --> 키워드검색
    의미검색 --> 재정렬
    키워드검색 --> 재정렬

    %% 유사성 분석 도구
    조사관 --> 발음변환
    발음변환 --> 발음비교
    조사관 --> 특징추출
    특징추출 --> 이미지비교
    조사관 --> 의미추출
    의미추출 --> 개념비교

    %% 데이터 소스
    상표DB --> 의미검색
    판례DB --> 의미검색
    상표DB --> 키워드검색
    판례DB --> 키워드검색

    %% 스타일링
    classDef userStyle fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef agentStyle fill:#fff8e1,stroke:#ff8f00,stroke-width:2px
    classDef reasonStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef searchStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef phoneticStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef visualStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef conceptStyle fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef dataStyle fill:#f5f5f5,stroke:#616161,stroke-width:2px

    class 입력,결과 userStyle
    class 분석관,조사관,심사관,작성관 agentStyle
    class 사고,생성 reasonStyle
    class 의미검색,키워드검색,재정렬 searchStyle
    class 발음변환,발음비교 phoneticStyle
    class 특징추출,이미지비교 visualStyle
    class 의미추출,개념비교 conceptStyle
    class 상표DB,판례DB dataStyle
```

---

## ⚙️ 시스템 아키텍처 (알고리즘 중심)

> 개발자를 위한 기술 스택 중심의 아키텍처입니다.

```mermaid
flowchart TB
    subgraph UI["🖥️ User Interface Layer"]
        Dashboard["Web/App Dashboard<br/>(React, Next.js)"]
    end

    subgraph Orchestration["🎯 Agent Orchestration Layer"]
        Controller["Agent Controller<br/>(LangGraph)"]
        
        subgraph Agents["Multi-Agent System"]
            QueryAgent["📋 Query Agent<br/>(질의 분석관)"]
            SearchAgent["🔍 Search Agent<br/>(조사관)"]
            JudgeAgent["⚖️ Judge Agent<br/>(심사관)"]
            SummaryAgent["📝 Summary Agent<br/>(작성관)"]
        end
    end

    subgraph Reasoning["🧠 Reasoning Core"]
        LLM["GPT-4o<br/>(Azure OpenAI)"]
        ReAct["ReAct + CoT<br/>Reasoning Loop"]
    end

    subgraph Retrieval["📚 Retrieval System (Advanced RAG)"]
        HybridSearch["Hybrid Search Engine"]
        VectorDB["Vector DB<br/>(Milvus/Pinecone)"]
        KeywordDB["Keyword Search<br/>(Elasticsearch)"]
        Reranker["Cross-Encoder<br/>Re-ranker"]
    end

    subgraph DomainModels["🤖 Domain-Specific Models"]
        subgraph Phonetic["호칭 유사성"]
            G2P["G2P 변환<br/>(KoG2P)"]
            PhoneticBERT["Phonetic-BERT<br/>(자소 단위)"]
        end
        
        subgraph Visual["외관 유사성"]
            FALDR["FALDR-Net<br/>(Frequency-Aware)"]
            DeepMetric["Deep Metric Learning<br/>(Siamese Network)"]
        end
        
        subgraph Conceptual["관념 유사성"]
            CLIP["CLIP<br/>(Cross-Modal)"]
        end
    end

    subgraph DataPipeline["⚙️ Data Pipeline"]
        Ingestion["Data Ingestion<br/>(Airflow, Spark)"]
        Chunking["Legal Structure-Aware<br/>Chunking"]
        Embedding["Embedding<br/>Generation"]
    end

    subgraph DataSources["📦 External Data Sources"]
        KIPRIS["KIPRIS API<br/>(상표 DB)"]
        CaseDB["판례 DB<br/>(대법원)"]
    end

    %% User Flow
    Dashboard --> Controller
    Controller --> QueryAgent
    QueryAgent --> SearchAgent
    SearchAgent --> JudgeAgent
    JudgeAgent --> SummaryAgent
    SummaryAgent --> Dashboard

    %% Agent to Tools
    Controller <--> LLM
    LLM <--> ReAct

    %% Search Agent Tools
    SearchAgent --> HybridSearch
    HybridSearch --> VectorDB
    HybridSearch --> KeywordDB
    VectorDB --> Reranker
    KeywordDB --> Reranker

    %% Domain Model Connections
    SearchAgent --> G2P
    G2P --> PhoneticBERT
    SearchAgent --> FALDR
    SearchAgent --> DeepMetric
    SearchAgent --> CLIP

    %% Data Flow
    KIPRIS --> Ingestion
    CaseDB --> Ingestion
    Ingestion --> Chunking
    Chunking --> Embedding
    Embedding --> VectorDB

    %% Styling
    classDef uiStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef agentStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef llmStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef ragStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef modelStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef dataStyle fill:#fff8e1,stroke:#f57f17,stroke-width:2px
    class Dashboard uiStyle
    class Controller,QueryAgent,SearchAgent,JudgeAgent,SummaryAgent agentStyle
    class LLM,ReAct llmStyle
    class HybridSearch,VectorDB,KeywordDB,Reranker ragStyle
    class G2P,PhoneticBERT,FALDR,DeepMetric,CLIP modelStyle
    class Ingestion,Chunking,Embedding,KIPRIS,CaseDB dataStyle
```

---

## 🔧 핵심 기술

### 상표 유사성 3요소 분석

| 요소 | 기술 | 설명 |
|:----:|:-----|:-----|
| **🔊 호칭** | G2P + Phonetic-BERT | 한국어 음운 변동을 반영한 발음 유사도 분석 |
| **👁️ 외관** | FALDR-Net + CLIP | 로고의 형태/색상/패턴 특징 기반 시각적 유사도 |
| **💡 관념** | CLIP + LLM | 브랜드가 주는 의미/인상의 유사성 판단 |

### RAG vs Fine-tuning

본 시스템은 **RAG (Retrieval-Augmented Generation)** 아키텍처를 채택합니다.

| 항목 | Fine-tuning | RAG (본 시스템) |
|:-----|:----------:|:---------------:|
| 지식 최신성 | ❌ 학습 시점 고정 | ✅ 실시간 업데이트 |
| 환각 위험 | ⚠️ 17~33% | ✅ 근거 기반 생성 |
| 설명 가능성 | ❌ 블랙박스 | ✅ 참조 문서 명시 |
| 비용 | 💰 GPU 학습 비용 | 💵 API 호출 비용 |

---

## 📚 문서

- [심층 연구 보고서](2026-01-09_deep_research.md) - 전체 기술 아키텍처 및 SOTA 연구 분석

---

## 📄 License

MIT License
