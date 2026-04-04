"""
ReAct-style QA agent with iterative retrieval.

Flow per question:
  1. Plan retrieval (classify query → choose paths + formulate query)
  2. Execute retrieval via UnifiedSearchService
  3. Assess sufficiency of results
  4. If insufficient and budget remains, refine and loop
  5. Generate final answer grounded in retrieved context
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from unified_memory.core.json_utils import validate_and_repair_json
from unified_memory.core.tokenizer import ContextWindowManager
from unified_memory.observability.tracing import traced

if TYPE_CHECKING:
    from unified_memory.retrieval.unified import UnifiedSearchService
    from unified_memory.llm.base import BaseLLMProvider
    from unified_memory.storage.sql.session_manager import ChatSessionManager
    from unified_memory.namespace.manager import NamespaceManager
    from unified_memory.llm.provider_registry import ProviderRegistry

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3


class QAAgent:
    """Multi-iteration retrieval + generation agent."""

    def __init__(
        self,
        search_service: "UnifiedSearchService",
        namespace_manager: "NamespaceManager",
        provider_registry: "ProviderRegistry",
        session_manager: Optional["ChatSessionManager"] = None,
    ) -> None:
        self.search_service = search_service
        self.namespace_manager = namespace_manager
        self.provider_registry = provider_registry
        self.session_manager = session_manager

    @traced("agent.answer")
    async def answer(
        self,
        question: str,
        namespace: str,
        user_id: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        all_results: list = []
        reasoning_trace: List[Dict[str, Any]] = []

        # Get Tenant LLM
        ns_cfg = await self.namespace_manager.get_config(namespace)
        tenant_cfg = None
        if ns_cfg:
            tenant_cfg = await self.namespace_manager.get_tenant_config(ns_cfg.tenant_id)

        llm = None
        if tenant_cfg and tenant_cfg.llm:
            try:
                llm = self.provider_registry.get_llm_provider(f"{tenant_cfg.llm.get('provider')}:{tenant_cfg.llm.get('model')}")
            except ValueError:
                pass
        
        if not llm:
            llm_providers = list(getattr(self.provider_registry, "_llm_providers", {}).values())
            if not llm_providers:
                raise ValueError("No LLM providers available")
            llm = llm_providers[0]

        # Determine which paths are actually available
        available_paths = await self._available_paths(namespace)

        total_usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}

        def record_usage(i_toks: int, o_toks: int, r_toks: int) -> None:
            total_usage["input_tokens"] += i_toks
            total_usage["output_tokens"] += o_toks  
            total_usage["reasoning_tokens"] += r_toks

        for iteration in range(MAX_ITERATIONS):
            plan = await self._plan_retrieval(
                question, all_results, iteration, available_paths, llm, record_usage
            )
            reasoning_trace.append(
                {"step": "plan", "iteration": iteration, "plan": plan}
            )

            query = plan.get("query", question)
            paths = plan.get("paths", available_paths)

            # Apply session doc filter if available
            doc_filter = None
            if session_id and self.session_manager:
                doc_ids = await self.session_manager.get_associated_documents(
                    session_id
                )
                if doc_ids:
                    doc_filter = {"document_id": doc_ids}

            results = await self.search_service.search(
                query=query,
                user_id=user_id,
                namespace=namespace,
                request_options={"paths": paths, "top_k": 10},
                filters=doc_filter,
            )
            all_results = self._merge_deduplicate(all_results, results)
            reasoning_trace.append(
                {"step": "retrieve", "query": query, "result_count": len(results)}
            )

            assessment = await self._assess_sufficiency(question, all_results, llm, record_usage)
            reasoning_trace.append({"step": "assess", "assessment": assessment})

            if assessment.get("sufficient") or assessment.get("confidence", 0) > 0.7:
                break

        # Generate final answer limiting context window
        llm_model = tenant_cfg.llm.get("model", "gpt-4o") if tenant_cfg and tenant_cfg.llm else "gpt-4o"
        context_mgr = ContextWindowManager(model=llm_model)
        fitted_results = context_mgr.fit_results(all_results)
        
        context = "\n\n".join(
            r.content for r in fitted_results if r.content
        )
        prompt_text = (
            f"Answer the following question based on the provided context. "
            f"Cite specific passages when possible. If the context does not "
            f"contain enough information, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        answer = await llm.generate(
            prompt_text,
            temperature=0.3,
            usage_callback=record_usage,
        )

        sources = [
            {
                "id": r.id,
                "score": r.score,
                "snippet": (r.content or ""),
            }
            for r in all_results
        ]
        
        retrieval_context = [
            {"id": r.id, "score": r.score, "snippet": (r.content or "")}
            for r in all_results
        ]

        logger.info(
            f"QA Token Usage: input={total_usage['input_tokens']} "
            f"output={total_usage['output_tokens']}"
        )

        return {
            "answer": answer,
            "sources": sources,
            "reasoning_trace": reasoning_trace,
            "retrieval_context": retrieval_context,
            "token_usage": total_usage,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _available_paths(self, namespace: str) -> List[str]:
        paths = []
        ns_cfg = await self.namespace_manager.get_config(namespace)
        tenant_config = None
        if ns_cfg:
            tenant_config = await self.namespace_manager.get_tenant_config(ns_cfg.tenant_id)

        if self.search_service.dense_retriever is not None:
            paths.append("dense")
        if self.search_service.sparse_retriever is not None:
            paths.append("sparse")
        if self.search_service.graph_retriever is not None and getattr(tenant_config, "enable_graph_storage", True):
            paths.append("graph")
        return paths or ["dense"]

    async def _plan_retrieval(
        self,
        question: str,
        existing_results: list,
        iteration: int,
        available_paths: List[str],
        llm: "BaseLLMProvider",
        usage_callback: Any = None,
    ) -> Dict[str, Any]:
        if iteration == 0:
            return {"query": question, "paths": available_paths}

        summary = "\n".join(
            f"- {r.content[:200]}" for r in existing_results[:15] if r.content
        )
        prompt = (
            f"Question: {question}\n"
            f"Previous retrieval returned:\n{summary}\n\n"
            f"The results were insufficient. Suggest a reformulated query and "
            f"which retrieval paths to use from {available_paths}.\n"
            f'Return JSON: {{"query": "...", "paths": [...]}}'
        )
        try:
            raw = await llm.generate(
                prompt, temperature=0.0, usage_callback=usage_callback
            )
            return validate_and_repair_json(raw, expected_keys=["query", "paths"])
        except Exception:
            logger.debug("Failed to parse retrieval plan, using defaults")
        return {"query": question, "paths": ["dense"]}

    async def _assess_sufficiency(
        self, question: str, results: list, llm: "BaseLLMProvider", usage_callback: Any = None
    ) -> Dict[str, Any]:
        if not results:
            return {"sufficient": False, "confidence": 0.0, "missing": "no results"}

        context = "\n".join(
            f"[{i}] (score={r.score:.2f}) {r.content[:200]}"
            for i, r in enumerate(results[:15])
            if r.content
        )
        prompt = (
            f"Question: {question}\n\nRetrieved passages:\n{context}\n\n"
            f"Assess: Can these passages answer the question? "
            f'Respond as JSON: {{"sufficient": true/false, "confidence": 0.0-1.0, '
            f'"missing": "what info is missing or empty string"}}'
        )
        try:
            raw = await llm.generate(
                prompt, temperature=0.0, max_output_tokens=200, usage_callback=usage_callback
            )
            return validate_and_repair_json(raw, expected_keys=["sufficient", "confidence"])
        except Exception:
            logger.debug("Failed to parse sufficiency assessment")

        # Heuristic fallback: if best score > 0.7 and we have 3+ results, assume OK
        if len(results) >= 3 and results[0].score > 0.7:
            return {"sufficient": True, "confidence": 0.75}
        return {"sufficient": False, "confidence": 0.3}

    @staticmethod
    def _merge_deduplicate(existing: list, new: list) -> list:
        """Merge results, deduplicating by canonical key."""
        from unified_memory.retrieval.fusion import canonical_key
        
        seen = {canonical_key(r) for r in existing}
        merged = list(existing)
        for r in new:
            key = canonical_key(r)
            if key not in seen:
                merged.append(r)
                seen.add(key)
        merged.sort(key=lambda x: x.score, reverse=True)
        return merged
