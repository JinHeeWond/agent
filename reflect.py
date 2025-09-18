#!/usr/bin/env python
# coding: utf-8

# In[21]:


# ==== 1번 셀: 중앙 라우터 방식 최종 코드 (모든 문제 해결 버전) ==

import os
import json
from typing import TypedDict, Annotated, List, Dict, Any
from IPython.display import Image, display
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

# ==========================================================
# ✨ 1. AgentState 및 전문가 노드/Tool 정의 ✨
# ==========================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    next_node: str
    response_type: str
    memory_summary: str
    clarification_count: int
    reflection_count: int 

  # === 컨텍스트 윈도우/서머리 예산 ===
CONTEXT_BUDGET_TOKENS = 8000       # 프롬프트에 실을 '최근 대화' 토큰 예산
SUMMARIZE_AT_TOKENS   = 12000      # 이 이상 커지면 오래된 구간을 요약해 memory_summary에 축약
SUMMARY_MAX_TOKENS    = 600        # 러닝 서머리 길이 목표(LLM에 요청)

# ===== Global I/O language settings =====
RESEARCH_LANG_MODE = "both"  # "en" | "ko" | "both"
OUTPUT_LANG = "en"           # "en" | "ko"



# ==== 환경 설정, 모델 로딩, Tool 정의 ====
def setup_environment():
    # 💥 중요: "tvly-..." 부분에 본인의 Tavily API 키를 입력하세요.
    TAVILY_API_KEY = "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"
    os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY
    return TavilyClient(api_key=TAVILY_API_KEY)

tavily_client = setup_environment()

# 🤫 제공해주신 본인의 Gemini API 키를 "" 안에 붙여넣으세요.
# (보안을 위해 이 키는 나중에 꼭 교체해주세요!)
GOOGLE_API_KEY = "AIzaSyAymOmcEDUuV9xWZ-o_3ecIkRrlR8P6I5E"

# ✨ 모델을 초기화할 때 위 API 키를 직접 전달합니다.
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY
)

json_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    model_kwargs={"response_mime_type": "application/json"},
    google_api_key=GOOGLE_API_KEY
)

# ✨ 헬퍼
import re
from difflib import SequenceMatcher
from urllib.parse import urlparse

WHITELIST_DOMAINS = {
    # 공공/관광청/대형 OTA/공식
    "japan-guide.com", "kyoto.travel", "japan.travel", "tokyometro.jp", "jr-odekake.net",
    "agoda.com", "booking.com", "hotels.com", "rakuten.com", "rakuten.co.jp",
    "jrpass.com", "japan-rail-pass.com",
}
BLACKLIST_DOMAINS = {
    # 과도한 스팸/스크래핑성(원하면 비워두세요)
}

def _norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_text(a), _norm_text(b)).ratio()

def _is_domain_ok(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
        host = ".".join(netloc.split(".")[-3:])  # 마지막 2~3 레벨 비교용
        if any(d in netloc for d in BLACKLIST_DOMAINS):
            return False
        # 화이트리스트가 비어있지 않으면 우선 가점: 미포함도 허용하되 정렬에서 밀림
        return True
    except Exception:
        return True

def _score_result(item: dict) -> float:
    """
    간단 스코어링: 화이트리스트 도메인 가점 + 검색엔진 점수 보정
    """
    base = float(item.get("score") or 0.0)
    url = item.get("url", "")
    netloc = urlparse(url).netloc.lower()
    wl_bonus = 0.2 if any(d in netloc for d in WHITELIST_DOMAINS) else 0.0
    return base + wl_bonus

def merge_and_dedup_results(*lists):
    """
    여러 검색 결과 리스트를 합치고 URL/제목 유사도로 중복 제거.
    """
    merged = []
    seen_urls = set()

    # 1) 단순 URL 중복 제거
    for lst in lists:
        for r in lst or []:
            url = r.get("url") or ""
            if not url or url in seen_urls:
                continue
            if not _is_domain_ok(url):
                continue
            seen_urls.add(url)
            merged.append(r)

    # 2) 제목 유사도 기반 중복 제거
    deduped = []
    for r in merged:
        title = r.get("title") or ""
        is_dup = False
        for kept in deduped:
            if _similar(title, kept.get("title") or "") > 0.9:
                is_dup = True
                break
        if not is_dup:
            deduped.append(r)

    # 3) 스코어 정렬(화이트리스트 가점)
    deduped.sort(key=_score_result, reverse=True)

    # 4) 상위 N개만 반환 (너무 많으면 생성 품질 저하)
    return deduped[:8]


def _approx_tokens(text: str) -> int:
    # 대략 4문자 ≈ 1토큰 가정 (모델 무관 근사치)
    if not text:
        return 0
    return max(1, len(text) // 4)

def _build_serialized_history(messages: List[BaseMessage]) -> str:
    """모든 메시지를 직렬화 (요약 트리거 계산용)"""
    lines = []
    for m in messages:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            if not getattr(m, "tool_calls", []):
                lines.append(f"Assistant: {m.content}")
    return "\n".join(lines)

def _get_conversation_history_windowed(state: AgentState, budget_tokens: int = CONTEXT_BUDGET_TOKENS) -> str:
    """
    memory_summary(러닝 서머리) + 최근 대화만 모아
    LLM에 넣을 컨텍스트 문자열을 만든다.
    """
    recent_blocks: List[str] = []
    used = 0
    # 최신부터 거꾸로 모으면서 예산 소진 시 중단
    for m in reversed(state["messages"]):
        # Human/AI만 최근 대화에 포함 (툴콜 내용은 제외)
        if isinstance(m, HumanMessage):
            s = f"User: {m.content}"
        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", []):
            s = f"Assistant: {m.content}"
        else:
            continue
        t = _approx_tokens(s)
        if used + t > budget_tokens:
            break
        recent_blocks.append(s)
        used += t

    recent_part = "\n".join(reversed(recent_blocks)).strip()
    summary_part = (state.get("memory_summary") or "").strip()
    if summary_part and recent_part:
        return f"[SUMMARY]\n{summary_part}\n\n[RECENT]\n{recent_part}"
    elif summary_part:
        return f"[SUMMARY]\n{summary_part}"
    else:
        return recent_part

from typing import Optional

def _maybe_update_memory_summary(state: AgentState) -> Optional[str]:
    """
    대화가 SUMMARIZE_AT_TOKENS를 넘으면,
    '오래된 구간'을 요약해 memory_summary에 누적한다.
    반환값: 갱신된 summary 문자열(없으면 None)
    """
    full = _build_serialized_history(state["messages"])
    if _approx_tokens(full) < SUMMARIZE_AT_TOKENS:
        return None  # 아직 여유 있음

    # 오래된 절반 정도를 요약 대상으로 잡는다
    # (필요하면 더 정교한 분할 로직으로 바꿔도 됨)
    midpoint = max(1, len(full) * 6 // 10)  # 앞쪽 60% 근사
    old_chunk = full[:midpoint]
    keep_chunk = full[midpoint:]  # 최근부

    prompt = f"""
당신은 대화 러닝 서머리 요약기입니다.
아래 '오래된 대화'를 읽고, 앞으로의 작업에 필요한 핵심만 한국어로 간결하게 요약하세요.
- 사용자 선호/제약(예산, 일정, 선호 스타일)
- 이미 내려진 결정/합의
- 해결된/미해결 이슈, 액션아이템
- 불필요한 잡담/중복 제거
- {SUMMARY_MAX_TOKENS} 토큰 이내 목표

[오래된 대화]
{old_chunk}
"""
    try:
        summary = model.invoke(prompt).content.strip()
    except Exception:
        summary = ""

    # 이전 요약과 합치기 (중복 최소화는 단순 연결로 처리)
    prev = (state.get("memory_summary") or "").strip()
    new_summary = (prev + "\n" + summary).strip() if summary else prev or ""

    # keep_chunk 자체는 상태(messages)에 그대로 남아있지만,
    # 프롬프트 구성은 windowed 함수가 담당하므로 문제 없음.
    return new_summary or None


def _get_conversation_history(state: AgentState) -> str:
    """메시지 기록 전체를 하나의 대화 문자열로 결합합니다. (장기 기억 역할)"""
    history = []
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            history.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            if not getattr(msg, 'tool_calls', []):
                history.append(f"Assistant: {msg.content}")
    return "\n".join(history)

def _get_last_user_message(state: AgentState) -> str:
    """가장 최근 Human 메시지의 content만 반환 (토픽 섞임 방지용)"""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

def is_ambiguous_with_llm(query: str) -> bool:
    """Judges whether the user's request is ambiguous using an LLM."""
    print(f"\n[LLM Check] 🤖 Is the conversation ambiguous?\n--- Conversation ---\n{query}\n--------------------")
    system_prompt = """
    You are an expert at judging the ambiguity of user requests.
    Based on the entire conversation history, you must determine if the information is specific enough to carry out the user's final goal.

    <Criteria>
    - A user's first question is ambiguous if it is too broad, lacking a subject, target, or purpose (e.g., "Write a report").
    - Even if the user has provided an answer, the request is still ambiguous if key information required to start the task is still missing.
    - If the conversation has progressed enough that a concrete plan can be created, it is no longer ambiguous.

    <Examples>
    - Conversation: "User: Write a report" -> Ambiguous (What is the topic?)
    - Conversation: "User: Write a report\nAssistant: What is the topic?\nUser: A marketing analysis report" -> Still ambiguous (What product? For whom?)
    - Conversation: "User: Write a marketing analysis report on the new robot vacuum sold on Coupang for the executive team." -> Specific (not ambiguous)

    Analyze the user's request and respond ONLY in the following JSON format.
    {"is_ambiguous": boolean, "reason": "The reason for your judgment."}
    """
    try:
        response = json_model.invoke(system_prompt + f"\n\nConversation History:\n\"{query}\"")
        content = response.content.strip()
        # 모델이 마크다운으로 응답할 경우를 대비한 전처리
        if content.startswith("```json"):
            content = content.strip("```json").strip("```").strip()
        result = json.loads(content)
        is_ambiguous = result.get("is_ambiguous", True)
        reason = result.get("reason", "No reason provided.")
        print(f"Decision: Ambiguous = {is_ambiguous} (Reason: {reason})")
        return is_ambiguous
    except Exception as e:
        print(f"Error during ambiguity check: {e}")
        return True  # On error, default to QA node for safety

def _generate_search_query_with_llm(plan_description: str, conversation_history: str, lang: str = "en") -> str:
    """
    Generate a short, search-ready query in the requested language.
    lang: "en" | "ko"
    """
    print(f"\n[LLM Check] ➡️ Optimizing search query ({lang}): '{plan_description}'")
    lang_clause = "in English" if lang == "en" else "in Korean"
    system_prompt = f"""
You convert a research step and chat history into a concise, high-intent search query {lang_clause}.
- Keep it SHORT (3~8 words).
- Include destination or key nouns if implied.
- Remove filler words.
- No quotes or extra commentary.
Return ONLY the query text.
"""
    response = model.invoke(system_prompt + f"\n\nPlan/Step: \"{plan_description}\"\nConversation: \"{conversation_history}\"")
    optimized_query = (response.content or "").strip()
    print(f"Decision: Optimized query ({lang}) = '{optimized_query}'")
    return optimized_query


def classify_intent_with_llm(query: str) -> str:
    """LLM을 사용하여 사용자의 요청 의도를 'search' 또는 'report'로 분류합니다."""
    print(f"\n[LLM Check] 🎯 Classifying intent for: '{query}'")
    system_prompt = """
    You are an expert at classifying user requests.
    Based on the user's request, you must classify the intent into one of two categories.

    <Categories>
    - 'search': The user wants to find or retrieve specific information, articles, facts, or data. They do not require a detailed, structured, or synthesized document.
    - 'report': The user wants a comprehensive, structured document that synthesizes information from multiple sources, such as a report, analysis, summary, plan, or detailed article.

    <Examples>
    - User Request: "최근 인플레이션에 대한 뉴스 기사를 찾아줘" -> 'search'
    - User Request: "토스 기업의 재무 성과에 대한 최신 기사들을 알려줘" -> 'search'
    - User Request: "2024년 한국 경제 동향에 대한 보고서를 작성해줘" -> 'report'
    - User Request: "챗GPT-4o와 Gemini 1.5 Pro의 성능을 비교 분석하는 문서를 만들어줘" -> 'report'
    - User Request: "한국의 인공지능 산업 발전 로드맵을 정리해줘" -> 'report'

    Analyze the user's request and respond ONLY with the category name ('search' or 'report').
    """
    response = model.invoke(system_prompt + f"\n\nUser Request:\n\"{query}\"")
    intent = response.content.strip().lower()
    print(f"Decision: Intent = {intent}")
    return intent

# --- [전체 코드-1] QA Tool 정의 ---
# --- [Part 1] QA Tool Definition ---
def generate_clarifying_question_tool(query: str) -> List[dict]:
    """Generates clarifying questions and options when the user's request is ambiguous."""
    print(f"\n[Tool Call] ❓ generate_clarifying_question_tool: '{query}'")
    system_prompt = """
    You are a proactive assistant that helps users accomplish their goals by gathering essential information.
    Your task is to identify key missing pieces of information in the user's request and ask concise, clear questions to get that information.
    The response must be in a specific JSON format. Do not add any extra text or explanations.
    
    <Criteria for a Good Clarifying Question>
    - The question should be specific, not general.
    - Provide at least 3 multiple-choice options for the user to select from.
    - Ask all questions in **English**.
    
    <Output Format>
    Return a JSON array of objects. Each object represents a question.
    [
      {"question": "What is the primary topic of your request?", "choices": ["Report on technology trends", "Analysis of market share", "Comparison of two products"]},
      {"question": "What is your preferred travel style?", "choices": ["Relaxed and leisurely", "Fast-paced and action-packed", "A mix of both"]},
      ...
    ]
    <Critical Requirements>
    - NO extra metadata, explanations, or wrapper objects.
    - Return pure JSON only.
    """
    user_prompt = f'userInput: "{query}"'
    try:
        response = json_model.invoke(
            system_prompt + "\n\n" + user_prompt
        )
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.strip("```json").strip("```").strip()
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON parsing failed: {e}")
        return [{"question": "What additional information can you provide to clarify your request?", "choices": ["Purpose", "Target audience", "Budget", "Duration"]}]
    except Exception as e:
        print(f"⚠️ Unexpected error occurred: {e}")
        return [{"question": f"An error occurred while processing your request. (Error: {e})", "choices": ["Yes", "No"]}]


# --- [전체 코드-2] Plan Tool 정의 ---
def create_plan_tool(query: str) -> List[Dict[str, Any]]:
    """명확해진 사용자 요청을 바탕으로, 구조화된 리서치 계획을 JSON 객체 배열로 생성합니다."""
    print(f"\n[Tool Call] 📝 create_plan_tool (Advanced): '{query}'")
    system_prompt = """
You are a planner that must always return a valid JSON array where each step is a separate object.
<planning_rules>
1. Break down the user request into clear, actionable steps for research.
2. Each step MUST be its own object in an array.
3. MAXIMUM 10 steps allowed - create high-level phases for complex tasks.
4. Steps should be logically ordered and atomic (one clear action per step).
5. Number steps sequentially starting from 1.
6. Set dependencies only when a step genuinely cannot start without another step completing.
</planning_rules>
<output_format>
Return ONLY a JSON array of step objects:
[
  {
    "id": 1,
    "title": "Clear, actionable step title",
    "description": "Detailed explanation of what needs to be done in this step",
    "status": "pending",
    "research": true,
    "dependencies": [],
    "complexity": 3
  }
]
</output_format>
<critical_requirements>
- NO extra metadata, explanations, or wrapper objects.
- Return pure JSON only.
</critical_requirements>
"""
    response = json_model.invoke(system_prompt + f"\n\nUser Request: \"{query}\"")
    try:
        content = response.content.strip()
        # 마크다운 래퍼 제거
        if content.startswith("```json"):
            content = content.strip("```json").strip("```")
        return json.loads(content)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"⚠️ Warning: create_plan_tool failed to return valid JSON: {e}")
        # 오류 발생 시 빈 리스트 대신 None을 반환하거나, 에러를 던져 상위 노드에서 처리하도록 할 수 있음
        return None 

# --- [전체 코드-3] Web Search Tool 정의 ---
def web_search_tool(query: str) -> List[Dict[str, Any]]:
    print(f"\n[Tool Call] 🔍 web_search_tool: '{query}'")
    try:
        search_results = tavily_client.search(query, search_depth="advanced", max_results=3)
        print(f"🔍 Tavily API Raw Response: {search_results}")
        return search_results.get('results', []) or []
    except Exception as e:
        print(f"⚠️ Tavily error: {e}")
        return []

# --- [전체 코드-4] Draft Tool 정의 ---
def generate_draft_tool(research_results: List[str], query: str) -> str:
    """사용자 요청과 리서치 결과를 바탕으로 최종 보고서 초안을 작성합니다."""
    print(f"\n[Tool Call] ✍️ generate_draft_tool: '{query}'")
    results_str = "\n\n".join(research_results)
    prompt = f"""
You are an expert report writer.
Based on the user's request and the following research results, write a detailed, structured draft.

<User Request>
{query}

<Research Results>
{results_str}

<Instructions>
- Write in Korean
- Use headings and bullet points if appropriate
- Be clear and concise
- Include factual information only from the research results
"""
    response = model.invoke(prompt)
    return response.content

def reflect_and_critique_tool(conversation: str) -> str:
    print(f"\n[Tool Call] 🤔 reflect_and_critique_tool...")
    system_prompt = """
    You are an expert critic and strategist. Your task is to review the final draft of a report or plan and provide a constructive critique and recommendations for improvement.

    <Instructions>
    - Analyze the content for clarity, completeness, logical flow, and alignment with the original user request.
    - Identify strengths and weaknesses of the draft.
    - Provide specific, actionable suggestions for improvement.
    - Write the entire critique in **English**.
    - Do not rewrite the draft itself, only provide the critique.

    Return only the critique and suggestions in a well-structured format with clear headings.
    """
    
    full_prompt = system_prompt + f"\n\nConversation History:\n\"{conversation}\""
    print("--- LLM Prompt for Critique Generation ---")
    print(full_prompt)
    print("---------------------------------")
    
    response = model.invoke(full_prompt)
    
    print("--- Original Critique Response from LLM ---")
    print(response.content)
    print("---------------------------------")

    return response.content

# ==========================================================
# ✨ 1. 전문가 노드 구현 ✨
# ==========================================================
# 수정된 qa_node 함수
def qa_node(state: AgentState):
    print("\n--- Node: ❓ QA Specialist ---")
    # ✅ 전체 대화 기록을 기반으로 질문 생성하도록 변경
    conversation_history = _get_conversation_history_windowed(state)
    questions = generate_clarifying_question_tool(conversation_history)
    return {
        "messages": [
            ToolMessage(
                name="generate_clarifying_question_tool",
                content=json.dumps(questions, ensure_ascii=False),
                tool_call_id="manual_qa"
            )
        ]
    }

def planner_node(state: AgentState):
    print("\n--- Node: 📝 Planning Specialist ---")
    # ✅ 토픽 섞임 방지: 마지막 사용자 메시지만 기반으로 플랜 생성
    last_user = _get_last_user_message(state)
    plan = create_plan_tool(last_user)
    cleaned = []
    for step in (plan or []):
        if isinstance(step, dict):
            step = {**step}
            # ✅ 기본적으로 research는 True 유지 (툴이 지정했으면 그 값 사용)
            step["research"] = bool(step.get("research", True))
            cleaned.append(step)
    return {
        "messages": [
            ToolMessage(
                name="create_plan_tool",
                content=json.dumps(cleaned, ensure_ascii=False),
                tool_call_id="manual_planner"
            )
        ]
    }

def researcher_node(state: AgentState):
    print("\n--- Node: 🔍 Research Specialist ---")

    # 최신 plan 추출
    plan = []
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage) and m.name == "create_plan_tool":
            try:
                plan = json.loads(m.content) or []
            except json.JSONDecodeError:
                plan = []
            break

    conversation_history = _get_conversation_history_windowed(state)
    all_results_en = []
    all_results_ko = []

    def do_search_once(desc: str, lang: str):
        q = _generate_search_query_with_llm(desc, conversation_history, lang=lang)
        print(f"🔎 Executing {lang.upper()} search for: '{q}' (step: '{desc[:60]}...')")
        return web_search_tool(q)  # returns list

    has_research_step = False
    if plan:
        for step in plan:
            if step.get("research"):
                has_research_step = True
                # === 하이브리드 검색 ===
                if RESEARCH_LANG_MODE in ("both", "en"):
                    res_en = do_search_once(step.get("description", ""), "en")
                    if res_en:
                        print(f"✅ EN results: {len(res_en)}")
                        all_results_en.extend(res_en)
                if RESEARCH_LANG_MODE in ("both", "ko"):
                    res_ko = do_search_once(step.get("description", ""), "ko")
                    if res_ko:
                        print(f"✅ KO results: {len(res_ko)}")
                        all_results_ko.extend(res_ko)

    # Fallback when no explicit research steps
    if plan and not has_research_step:
        seed = " ".join([str(s.get("title", "")) for s in plan[:3]]).strip() or conversation_history[-200:]
        if RESEARCH_LANG_MODE in ("both", "en"):
            all_results_en.extend(do_search_once(seed, "en"))
        if RESEARCH_LANG_MODE in ("both", "ko"):
            all_results_ko.extend(do_search_once(seed, "ko"))

    if not plan:  # No plan at all
        seed = conversation_history
        if RESEARCH_LANG_MODE in ("both", "en"):
            all_results_en.extend(do_search_once(seed, "en"))
        if RESEARCH_LANG_MODE in ("both", "ko"):
            all_results_ko.extend(do_search_once(seed, "ko"))

    # === 결과 병합/중복제거 ===
    combined = merge_and_dedup_results(all_results_en, all_results_ko)
    combined_results_json_str = json.dumps(combined, ensure_ascii=False)

    return {
        "messages": [
            ToolMessage(
                content=combined_results_json_str,
                name="web_search_tool",
                tool_call_id="manual_researcher"
            )
        ]
    }


def generator_node(state: AgentState):
    print("\n--- Node: ✍️ Generation Specialist ---")
    conversation_history = _get_conversation_history_windowed(state)
    research_results = []
    plan = []

    found_research = False
    found_plan = False
    for m in reversed(state["messages"]):
        if not found_research and isinstance(m, ToolMessage) and m.name == "web_search_tool":
            try:
                raw_results = json.loads(m.content)
                if isinstance(raw_results, list):
                    research_results = raw_results
            except (json.JSONDecodeError, TypeError):
                research_results = []
            found_research = True

        if not found_plan and isinstance(m, ToolMessage) and m.name == "create_plan_tool":
            try:
                raw_plan = json.loads(m.content)
                if isinstance(raw_plan, list):
                    plan = raw_plan
            except (json.JSONDecodeError, TypeError):
                plan = []
            found_plan = True

        if found_research and found_plan:
            break

    results_str = "\n\n".join(
        [f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nContent: {r.get('content', 'N/A')}" for r in research_results]
    )

    # 검색 결과가 비어있을 경우
    if not results_str.strip():
        if not plan or not any(step.get("research") for step in plan):
            response_content = "I'm sorry, I couldn't find any search results for your request. Please try again with different keywords."
            response_type = "search_summary"
        else:
            response_content = "I'm sorry, I couldn't find enough information to create a plan based on your request. Please provide more specific details, and I will try again."
            response_type = "report_draft"
        return {
            "messages": [AIMessage(content=response_content)],
            "response_type": response_type
        }

    # 결과가 있을 때
    if not plan or not any(step.get("research") for step in plan):
        prompt = f"""
        You are an expert at summarizing search results.
        Based on the user's request and the following research results, provide a concise and factual summary in **English**. Do not add any new information.
        <User Request>
        {conversation_history}
        <Research Results>
        {results_str}
        """
        response_type = "search_summary"
    else:
        prompt = f"""
        You are an expert report writer.
        Based on the user's request and the following research results, write a detailed, structured draft in **English**.
        The draft should be a concrete plan or report, not just a summary.
        <User Request>
        {conversation_history}
        <Research Results>
        {results_str}
        <Instructions>
        - Write a detailed travel plan, structured by day (Day 1, Day 2, etc.) when applicable.
        - Include specific attractions, restaurants, and activities when the topic is travel.
        - Ensure the plan aligns with the user's constraints provided in the conversation history.
        - Use headings and bullet points to make the plan easy to read.
        - Be clear and concise.
        - Include factual information only from the research results.
        - If certain information is missing (e.g., a specific festival date), state this clearly and suggest a general plan.
        </Instructions>
        """
        response_type = "report_draft"

    response = model.invoke(prompt)
    state_update = {
        "messages": [AIMessage(content=response.content)],
        "response_type": response_type
    }
    return state_update

def reflect_node(state: AgentState):
    print("\n--- Node: 🤔 Reflection Specialist ---")
    
    # 1. 비평 생성
    conversation_history = _get_conversation_history_windowed(state)
    critique = reflect_and_critique_tool(conversation_history)
    
    # 2. 비평을 바탕으로 재실행 필요 여부 판단
    reflection_judgment = model.invoke(
        f"""
        아래는 에이전트가 생성한 최종 보고서에 대한 비평입니다.
        이 비평이 중대한 오류나 부족함을 지적하여 보고서를 처음부터 다시 작성하거나, 중요한 정보를 추가로 검색해야 할 필요가 있다고 판단되면 'True',
        단순한 제안이나 작은 개선점에 대한 내용이라 현재 상태로도 충분하다고 판단되면 'False'를 반환하세요.
        오직 'True' 또는 'False'만 반환해야 합니다.
        
        <비평 내용>
        {critique}
        """
    ).content.strip().lower()

    should_re_run = reflection_judgment == "true"
    
    final_response = f"## 최종 보고서 초안:\n{state['messages'][-1].content}\n\n## 검토 및 제언:\n{critique}"
    new_count = state.get("reflection_count", 0) + 1

    # 3. 다음 노드로 전달할 정보 포함
    return {
        "messages": [AIMessage(content=final_response)],
        "response_type": "reflection_complete",
        "should_re_run": should_re_run,  # 이 정보를 `router`에 전달
        "reflection_count": new_count 
    }



# ==========================================================
# ✨ 2. 중앙 라우터(PM) 노드 및 그래프 조립 ✨
# ==========================================================
def router_node(state: AgentState):
    print("\n--- Node: 🧑‍💼 Central Router (PM) ---")
    msgs = state["messages"]

    # --- 1. 러닝 서머리 갱신 ---
    updates = {}
    try:
        new_sum = _maybe_update_memory_summary(state)
        if new_sum is not None:
            updates["memory_summary"] = new_sum
    except Exception as e:
        print(f"[summary] skip due to error: {e}")


    # --- 3. 리플렉션 단계에서 돌아온 경우 처리 ---
    if state.get("response_type") == "reflection_complete":
        should_re_run = state.get("should_re_run", False)
        reflection_count = state.get("reflection_count", 0)

        if reflection_count >= 2:  # ✨ 최대 2번까지만 허용
            print("➡️ Reflect 횟수 제한 도달. 종료합니다.")
            return {**updates, "next_node": "end"}

        # router_node (reflect 분기)
        if should_re_run:
            print("➡️ Critique suggests re-run. Routing back to 'planner'.")
            return {**updates, "next_node": "planner"}
        else:
            print("➡️ Critique suggests the draft is sufficient. Ending the conversation.")
            return {**updates, "next_node": "end"}



    # --- 4. 사용자로부터 새로운 요청을 받은 경우 처리 ---
    if isinstance(msgs[-1], HumanMessage):
        conversation_history = _get_conversation_history_windowed(state)
        if state.get("clarification_count", 0) >= 2:
            print("➡️ Clarification limit reached. Proceeding with the request.")
            return {**updates, "next_node": "planner"}
        if is_ambiguous_with_llm(conversation_history):
            updates["clarification_count"] = state.get("clarification_count", 0) + 1
            return {**updates, "next_node": "qa"}

    # --- 5. 도구(Tool) 호출 결과 처리 ---
    last = msgs[-1]
    if isinstance(last, ToolMessage):
        if last.name == "generate_clarifying_question_tool":
            return {**updates, "next_node": "end"}
        elif last.name == "create_plan_tool":
            return {**updates, "next_node": "researcher"}
        elif last.name == "web_search_tool":
            return {**updates, "next_node": "generator"}
    
    # --- 6. 생성된 최종 응답(AIMessage) 처리 (⭐수정된 핵심 부분⭐) ---
    if isinstance(last, AIMessage):
        # generator 노드가 최종 답변을 생성했을 때만 reflect 노드로 이동
        if state.get("response_type") in ["search_summary", "report_draft"]:
            print("➡️ 최종 보고서가 생성되었습니다. 'reflect' 노드로 라우팅합니다.")
            return {**updates, "next_node": "reflect"}

    return {**updates, "next_node": "end"}

workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("qa", qa_node)
workflow.add_node("planner", planner_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("generator", generator_node)
workflow.add_node("reflect", reflect_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda state: state["next_node"],
    {
        "qa": "qa",
        "planner": "planner",
        "researcher": "researcher",
        "generator": "generator",
        "reflect": "reflect",
        "end": END
    }
)

# QA 노드를 제외한 모든 전문가는 일을 마치면 다시 Router에게 보고
workflow.add_edge("planner", "router")
workflow.add_edge("researcher", "router")
workflow.add_edge("generator", "router")
workflow.add_edge("reflect", "router")

# QA 노드는 사용자 답변을 기다리기 위해 라우팅 없이 바로 종료
workflow.add_edge("qa", END)
# reflect 노드도 최종 답변 후 종료
#workflow.add_edge("reflect", END)

memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)
print("\n✅ Central Router Agent compiled successfully!")

try:
    image_bytes = app.get_graph().draw_png()
    display(Image(data=image_bytes))
except Exception as e:
    print(f"Could not generate graph image: {e}")


# In[22]:


# ==== 중복/유사 질의 감지 유틸 ====
import json, hashlib, difflib, re

asked_q_signatures = set()
# Clarifying 질문 라운드 카운터 (thread_id별)
clarify_rounds = {}
CLARIFY_ROUND_CAP = 2  # 최대 2회

asked_q_hashes = set()
asked_q_keys = []  # 유사도 비교용 원문(정규화 키)



def _normalize_qpayload(payload):
    def norm_text(t):
        t = t.lower()
        t = re.sub(r"\s+", " ", t).strip()
        t = re.sub(r"[^\w\s]", "", t)  # 문장부호 제거
        return t

    if isinstance(payload, list):
        items = []
        for q in payload:
            qt = norm_text(q.get("question", ""))
            choices = [norm_text(c) for c in q.get("choices", [])]
            choices = sorted(set(choices))
            items.append((qt, tuple(choices)))
        # 질문 순서 영향 없게 정렬
        items.sort()
        return items
    else:
        qt = norm_text(payload.get("question", ""))
        choices = [norm_text(c) for c in payload.get("choices", [])]
        choices = sorted(set(choices))
        return [(qt, tuple(choices))]

def is_duplicate_or_similar_questions(questions_json_str: str, similarity=0.9) -> bool:
    try:
        payload = json.loads(questions_json_str)
    except json.JSONDecodeError:
        return False

    items = _normalize_qpayload(payload)
    key = json.dumps(items, ensure_ascii=False)  # 유사도 비교용
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()

    # 1) 완전 동일
    if h in asked_q_hashes:
        return True

    # 2) 유사 (키 문자열끼리 비교)
    for prev_key in asked_q_keys:
        if difflib.SequenceMatcher(None, prev_key, key).ratio() >= similarity:
            return True

    # 3) 신규 등록
    asked_q_hashes.add(h)
    asked_q_keys.append(key)
    return False



# ==== 2nd Cell: Execution code for 'Expert Team Workflow' ====
# New conversation config
config = {"configurable": {"thread_id": "workflow-convo-200011911"}}

print("🤖 Agent: Hello! What can I help you with? (type 'quit' to exit)")

while True:
    user_input = input("🧑‍💻 User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("🤖 Agent: Ending the conversation.")
        break

    # 'app' is already defined in the notebook, so we use it directly
    events = app.stream(
        {"messages": [HumanMessage(content=user_input)]}, config, stream_mode="values"
    )

    final_message = None

    print("\n--------------------")
    for event in events:
        last_message = event["messages"][-1]

        if isinstance(last_message, ToolMessage):
            print(f"🤖 Agent: (Processing step: {last_message.name}...)")
        
        final_message = last_message

    if isinstance(final_message, AIMessage):
        print("\n[Final Answer]")
        print("🤖 Agent:", final_message.content)
    elif isinstance(final_message, ToolMessage) and final_message.name == "generate_clarifying_question_tool":
    # ===== (Optional) Clarifying round cap by thread_id =====
    # Requires: clarify_rounds = {}; CLARIFY_ROUND_CAP = 2  (defined above the loop)
        tid = config["configurable"].get("thread_id", "default-thread")
        cnt = clarify_rounds.get(tid, 0)
        if cnt >= CLARIFY_ROUND_CAP:
            print("🤖 Agent: Clarifying questions limit reached. Proceeding with best effort.")
            print("--------------------")
            continue
        else:
            clarify_rounds[tid] = cnt + 1

    # ===== Duplicate / Similarity guard =====
    # Requires: asked_q_signatures, is_duplicate_or_similar_questions(...) defined above the loop
        if is_duplicate_or_similar_questions(final_message.content):
            print("🤖 Agent: (Skipping repeated or highly similar clarifying questions)")
            print("--------------------")
            continue

    # ===== Render clarifying questions (English UI) =====
        try:
            questions_list = json.loads(final_message.content)
        except json.JSONDecodeError:
            # If payload is malformed, skip gracefully
            print("🤖 Agent: (Received invalid clarifying questions payload; skipping)")
            print("--------------------")
            continue

        print("🤖 Agent: I need to clarify a few things before proceeding. Please answer the following questions:")

        # List payload (preferred)
        if isinstance(questions_list, list):
            for question_data in questions_list:
                question_text = question_data.get("question", "")
                choices_list = question_data.get("choices", [])
                if question_text and isinstance(choices_list, list) and choices_list:
                    print(f"\n❓ {question_text}")
                    for i, choice in enumerate(choices_list, start=1):
                        print(f"   {i}. {choice}")

        # Single-object payload (fallback)
        elif isinstance(questions_list, dict):
            question_text = questions_list.get("question", "")
            choices_list = questions_list.get("choices", [])
            if question_text and isinstance(choices_list, list) and choices_list:
                print(f"\n❓ {question_text}")
                for i, choice in enumerate(choices_list, start=1):
                    print(f"   {i}. {choice}")

        else:
            # Unknown shape; show raw for debugging but stay user-friendly
            print("\n❓ (Unable to parse questions in a standard format.)")

        print("\n(Please provide your answers in the next User prompt.)")
        print("--------------------")


