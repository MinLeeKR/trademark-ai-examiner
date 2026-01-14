# 상표권 검색 및 유사도 분석 시스템 구축 전략

## 1. 개요 및 결론

**결론부터 말씀드리면, 질문자님의 접근 방식(검색 후 재순위화, Search-then-Rerank)이 전체 상표 공보를 Vector DB로 구축하는 방식보다 현업의 요구사항에 더 부합하며 실현 가능성이 높습니다.**

이를 전문 용어로는 **"Agentic RAG"** 또는 **"Two-Stage Retrieval"** 패턴이라고 합니다.
전체 데이터를 Vector DB화 할 경우 다음과 같은 치명적인 단점이 존재합니다.
1.  **데이터 신뢰성 문제:** 상표의 법적 상태(출원, 공고, 등록, 거절, 포기 등)는 매일 변합니다. 자체 DB는 이를 실시간 동기화하기 어렵습니다.
2.  **법적 분류 체계 무시:** 상표 검색은 단순한 '이미지 유사도'가 아니라, **지정상품(Nice 분류)**과 **비엔나 코드(도형 분류)**라는 법적 메타데이터가 핵심 필터링 조건이 되어야 합니다.

따라서, **"KIPRIS API로 광범위한 후보군(Recall)을 확보하고, AI 모델로 정밀하게 심사(Precision)하는 구조"**가 가장 합리적입니다.

---

## 2. 제안 아키텍처 (Search-then-Rerank)

이 시스템은 크게 **전처리(Query Expansion)** -> **검색(Retrieval)** -> **재순위화(Reranking)** 3단계로 구성됩니다.

### Step 1: 질의 확장 및 전처리 (Query Processing)

사용자가 입력한 상표명과 이미지를 검색 가능한 형태(키워드, 코드)로 변환합니다.

1.  **텍스트 (상표명):**
    *   **음운 유사 확장:** 한글의 자모를 분해하여 발음이 유사한 키워드를 생성합니다.
    *   **의미 유사 확장:** LLM을 사용하여 상표의 관념(의미)이 유사한 단어나 영어 번역어를 생성합니다.
2.  **이미지 (도형상표):**
    *   **비엔나 코드 예측:** AI 모델(CNN/ViT)을 사용하여 입력 이미지가 어떤 비엔나 코드(예: 3.4.1 - 소)에 해당하는지 예측합니다.

### Step 2: 후보군 검색 (Retrieval)

확장된 키워드와 코드를 사용하여 외부 공신력 있는 DB(KIPRIS)에서 후보군을 가져옵니다.

*   **Source:** [KIPRIS Plus API](https://plus.kipris.or.kr/)
*   **검색 전략:**
    *   (원본 상표명) AND (유사 지정상품 코드)
    *   (음운 유사 상표명) AND (유사 지정상품 코드)
    *   (예측된 비엔나 코드) AND (유사 지정상품 코드)
*   **목표:** 놓치는 상표가 없도록(High Recall) 50~100개 정도의 넉넉한 후보군을 수집합니다.

### Step 3: 유사도 판단 및 랭킹 (Reranking & Analysis)

수집된 후보군을 실제 심사 기준(외관, 칭호, 관념)에 맞춰 정밀 분석합니다.

*   **LLM 심사관 (AI Judge):** 수집된 텍스트/메타데이터 정보를 바탕으로 LLM(GPT-4, Claude 3.5 Sonnet 등)이 상표 심사 기준에 따른 유사 판단을 수행합니다.
*   **Visual Encoder:** 이미지 간의 순수 시각적 유사도(Pixel-level similarity)를 측정하기 위해 Vision Encoder(CLIP 등)를 보조적으로 사용합니다.

---

## 3. 세부 구현 가이드 및 기술 요소

### 3.1. 텍스트 음운 유사도 (Phonetic Similarity)

한글 상표는 **'발음'**의 유사성이 매우 중요합니다 (칭호 유사). 단순한 텍스트 거리(Levenshtein)보다는 **자소 분해(Jamo Decomposition)** 후 비교해야 합니다.

*   **알고리즘:** 한글 유니코드를 초성, 중성, 종성으로 분리하여 비교.
*   **라이브러리:** Python의 `jamo` 또는 `hangeul-jamo` 라이브러리 활용.
*   **참고 자료:**
    *   [Python Jamo Library Documentation](https://python-jamo.readthedocs.io/)
    *   [GeeksforGeeks: Phonetic Search Algorithms](https://www.geeksforgeeks.org/machine-learning/implement-phonetic-search-in-python-with-soundex-algorithm/) (개념 참조)

### 3.2. 비엔나 코드 자동 분류 (Vienna Code Classification)

이미지를 바로 Vector Search 하는 것보다, 이미지를 **비엔나 코드(법적 분류)**로 변환하여 검색하는 것이 훨씬 정확합니다.

*   **접근법:** 대규모 상표 이미지와 비엔나 코드가 태깅된 데이터셋(KIPRIS Bulk 데이터 등 활용)으로 Multi-label Image Classification 모델을 학습시킵니다.
*   **모델 구조:** ResNet50, EfficientNet, 또는 ViT(Vision Transformer).
*   **선행 연구:** WIPO나 EUIPO에서도 딥러닝을 이용한 비엔나 코드 분류 자동화를 연구하고 있습니다.
    *   [Deep Learning for identification of figurative elements in trademark images (DIVA Portal)](https://www.diva-portal.org/smash/record.jsf?pid=diva2:1606039)
    *   [Clarivate Trademark Vision](https://clarivate.com/intellectual-property/brand-ip-solutions/trademark-vision/) (상용 솔루션 사례)

### 3.3. 유사도 랭킹 (LLM Reranking)

단순 Cosine Similarity가 아니라, **"왜 유사한지"** 설명할 수 있어야 합니다.

*   **Prompt Engineering:** LLM에게 "상표 심사 기준"을 System Prompt로 주입합니다.
    *   *"당신은 특허청 심사관입니다. 다음 두 상표의 외관, 칭호, 관념을 비교하여 혼동 가능성을 0~100점 사이로 평가하고 그 이유를 설명하세요."*
*   **참고 기술:** [Rerankers and Two-Stage Retrieval (Pinecone)](https://www.pinecone.io/learn/series/rag/rerankers/)

---

## 4. 요약

| 구분 | 질문자님 아이디어 (Search-then-Rerank) | Full Vector DB 구축 (Naive RAG) |
| :--- | :--- | :--- |
| **데이터 최신성** | **높음** (KIPRIS 실시간 조회) | 낮음 (주기적 업데이트 비용 발생) |
| **구축 비용** | **낮음** (API 활용, 경량 모델) | 높음 (전체 이미지/텍스트 임베딩 저장) |
| **검색 정확도** | **높음** (비엔나 코드, 지정상품 필터링 활용) | 낮음 (법적 메타데이터 무시 경향) |
| **설명 가능성** | **높음** (LLM이 심사 논리 생성 가능) | 낮음 (단순 수치 기반) |

**결론적으로, 질문자님이 구상하신 프로세스가 실제 리걸테크(LegalTech) 서비스들이 채택하는 가장 합리적인 방식입니다.**
상표 검색은 '비슷하게 생긴 것'을 찾는 게 아니라 **'법적으로 혼동을 줄 수 있는 것'**을 찾는 것이기 때문입니다.
