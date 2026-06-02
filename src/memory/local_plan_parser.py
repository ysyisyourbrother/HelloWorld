#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local regex/heuristic parsing for agentic plan steps (tool name + arguments)."""

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

ToolSelection = Tuple[Optional[str], Dict[str, Any]]

_QUOTE_MARK = re.compile(r"['\"\u2018\u2019\u201c\u201d`]")
_RE_QUOTED = re.compile(
    r"'([^']*)'"
    r"|\"([^\"]*)\""
    r"|\u2018([^\u2019]*)\u2019"
    r"|\u201c([^\u201d]*)\u201d"
)
_RE_WHOLE_VIDEO = re.compile(r"\b(whole|entire|all|total)\b", re.IGNORECASE)
_RE_BETWEEN = re.compile(
    r"\bbetween\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s\b)?\s+and\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s\b)?",
    re.IGNORECASE,
)
_RE_FROM_TO = re.compile(
    r"\bfrom\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s\b)?\s+to\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s\b)?",
    re.IGNORECASE,
)
_RE_UNIFORM_SAMPLE = re.compile(r"\b(uniform(?:ly)?|sample(?:d|s)?)\b", re.IGNORECASE)
_RE_SUBTITLE_SCOPE = re.compile(r"\b(within|scope)\b", re.IGNORECASE)
_RE_CONTENT_ABOUT = re.compile(r"\bcontent about\b", re.IGNORECASE)
_DEFAULT_EVENT_SCOPE = [-60, 60]


class LocalPlanParser:
    """Parse [Scope]/[Search] plan text into (tool_name, arguments) like cloud tool_calls."""

    @staticmethod
    def _empty() -> ToolSelection:
        return None, {}

    @staticmethod
    def has_any_quote(text: str) -> bool:
        return _QUOTE_MARK.search(str(text or "")) is not None

    @staticmethod
    def extract_quoted_strings(text: str) -> List[str]:
        out: List[str] = []
        for match in _RE_QUOTED.finditer(str(text or "")):
            chunk = next((g for g in match.groups() if g is not None), "")
            word = str(chunk).strip()
            if word:
                out.append(word)
        return out

    @staticmethod
    def _clamp_keywords(items: Sequence[str], limit: int = 4) -> List[str]:
        uniq: List[str] = []
        seen = set()
        for raw in items:
            word = str(raw).strip()
            if not word:
                continue
            key = word.lower()
            if key in seen:
                continue
            seen.add(key)
            uniq.append(word)
            if len(uniq) >= limit:
                break
        return uniq

    @staticmethod
    def _parse_period_range(
        step_text: str, duration: float
    ) -> Optional[Tuple[float, float]]:
        text = str(step_text or "")
        match = _RE_BETWEEN.search(text)
        if match is None:
            match = _RE_FROM_TO.search(text)
        if match is None:
            return None
        start_t = float(match.group(1))
        end_t = float(match.group(2))
        if end_t < start_t:
            start_t, end_t = end_t, start_t
        end_cap = float(duration) if duration > 0.0 else end_t
        end_t = min(end_t, end_cap)
        start_t = max(0.0, start_t)
        return start_t, end_t

    @staticmethod
    def _extract_content_about_event(text: str) -> Optional[str]:
        text = str(text or "")
        match = _RE_CONTENT_ABOUT.search(text)
        if match is None:
            return None
        tail = str(text[match.end() :]).strip()
        if not tail:
            return None
        pairs = (
            ("'", "'"),
            ('"', '"'),
            ("\u2018", "\u2019"),
            ("\u201c", "\u201d"),
        )
        opener = tail[0]
        closer = None
        for open_ch, close_ch in pairs:
            if opener == open_ch:
                closer = close_ch
                break
        if closer is None:
            return None
        if opener in ("'", "\u2018"):
            body = tail[1:]
            close_idx = body.rfind(closer)
            if close_idx < 0:
                return None
            phrase = body[:close_idx].strip()
        else:
            close_idx = tail.find(closer, 1)
            if close_idx < 0:
                return None
            phrase = tail[1:close_idx].strip()
        return phrase or None

    @staticmethod
    def _try_parse_scope_event(step_text: str) -> ToolSelection:
        text = str(step_text or "")
        if _RE_CONTENT_ABOUT.search(text) is None:
            return LocalPlanParser._empty()
        event = LocalPlanParser._extract_content_about_event(text)
        if not event:
            return LocalPlanParser._empty()
        return (
            "_get_subset_by_event_frame",
            {"event": event, "scope": list(_DEFAULT_EVENT_SCOPE)},
        )

    @staticmethod
    def _try_parse_scope_period(
        step_text: str, duration: float
    ) -> ToolSelection:
        text = str(step_text or "")
        event_tool = LocalPlanParser._try_parse_scope_event(step_text)
        if event_tool[0]:
            return event_tool

        if LocalPlanParser.has_any_quote(text):
            return LocalPlanParser._empty()

        if _RE_WHOLE_VIDEO.search(text):
            end_t = float(duration) if duration > 0.0 else 0.0
            return (
                "_get_subset_by_period",
                {"start_time": 0.0, "end_time": end_t},
            )

        period = LocalPlanParser._parse_period_range(text, duration)
        if period is not None:
            start_t, end_t = period
            return (
                "_get_subset_by_period",
                {"start_time": start_t, "end_time": end_t},
            )

        return LocalPlanParser._empty()

    @staticmethod
    def try_parse_scope1_locally(
        step_text: str,
        duration: float,
        topk: int = 5,
    ) -> ToolSelection:
        del topk
        return LocalPlanParser._try_parse_scope_period(step_text, duration)

    @staticmethod
    def try_parse_scope1_locally_replan(
        step_text: str,
        duration: float,
        topk: int = 5,
    ) -> ToolSelection:
        del topk
        return LocalPlanParser._try_parse_scope_period(step_text, duration)

    @staticmethod
    def try_parse_scope2_locally(step_text: str) -> ToolSelection:
        quoted = LocalPlanParser.extract_quoted_strings(step_text)
        keywords = LocalPlanParser._clamp_keywords(quoted, limit=4)
        if len(keywords) < 1:
            return LocalPlanParser._empty()
        return ("_get_subset_by_keyword", {"keywords": keywords})

    @staticmethod
    def try_parse_search1_locally(
        step_text: str,
        topk: int = 5,
    ) -> ToolSelection:
        text = str(step_text or "")
        budget = int(topk) if int(topk) > 0 else 5

        if not LocalPlanParser.has_any_quote(text):
            if _RE_UNIFORM_SAMPLE.search(text):
                return ("_search_frames_just_by_scope", {"budget": budget})

        quoted = LocalPlanParser.extract_quoted_strings(text)
        entities = LocalPlanParser._clamp_keywords(quoted, limit=4)
        if entities:
            return (
                "_search_frames_with_multiple_entities",
                {"entities": entities, "top_k": budget},
            )

        return LocalPlanParser._empty()

    @staticmethod
    def try_parse_search1_locally_replan(
        step_text: str,
        topk: int = 5,
    ) -> ToolSelection:
        return LocalPlanParser.try_parse_search1_locally(step_text, topk=topk)

    @staticmethod
    def try_parse_search2_locally(step_text: str) -> ToolSelection:
        text = str(step_text or "")

        if not LocalPlanParser.has_any_quote(text):
            if _RE_SUBTITLE_SCOPE.search(text):
                return ("_search_subtitles_just_by_scope", {})

        quoted = LocalPlanParser.extract_quoted_strings(text)
        keywords = LocalPlanParser._clamp_keywords(quoted, limit=4)
        if keywords:
            return (
                "_search_subtitles_with_multiple_keywords",
                {"keywords": keywords},
            )

        return LocalPlanParser._empty()

if __name__ == "__main__":
    scope_plan1 = "[Scope] Pay attention to the entire video content."
    scope_plan2 = "[Scope] Pay attention to the content from 20 to 30 second."
    scope_plan3 = "[Scope] Pay attention to the content about 'Whitehead's first motorized flight'."
    search_plan1 = "[Search] Uniformly sample frames within this scope."
    search_plan2 = "[Search] Get subtitles related to 'sponsored', 'competition', 'presented'."
    search_plan3 = "[Search] Get frames about 'white team jersey number 13'."
    search_plan4 = "[Search] Get subtitles related to '13' or 'thirteen'."
    search_replan1 = "[Search] Get subtitles related to '13' or 'thirteen'."
    print("*"*6+"Origin"+"*"*6)
    print(scope_plan1)
    print(scope_plan2)
    print(scope_plan3)
    print(search_plan1)
    print(search_plan2)
    print(search_plan3)
    print(search_plan4)
    print("*"*6+"After Parse"+"*"*6)
    print(LocalPlanParser.try_parse_scope1_locally(scope_plan1, 1120.6))
    print(LocalPlanParser.try_parse_scope1_locally(scope_plan2, 1120.6))
    print(LocalPlanParser.try_parse_scope1_locally(scope_plan3, 3169.0))
    print(LocalPlanParser.try_parse_search1_locally(search_plan1))
    print(LocalPlanParser.try_parse_search2_locally(search_plan2))
    print(LocalPlanParser.try_parse_search1_locally(search_plan3))
    print(LocalPlanParser.try_parse_search2_locally(search_plan4))