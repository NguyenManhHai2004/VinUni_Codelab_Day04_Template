"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm & dịch vụ VinFast, Vinpearl và hỗ trợ khách hàng.
- Giọng nói: Chuyên nghiệp, thân thiện, chính xác, không bịa thông tin.

## AVAILABLE TOOLS
{tools}
- search_product_catalog: Tra cứu sản phẩm/dịch vụ theo danh mục và giá tối đa.
- submit_support_ticket: Tạo ticket hỗ trợ khách hàng khi có sự cố/khiếu nại.

## CORE RULES
1. KHÔNG BAO GIỜ bịa dữ liệu sản phẩm (giá, tính năng, tồn kho). Luôn gọi tool
   `search_product_catalog` để lấy dữ liệu thực.
2. KHÔNG BAO GIỜ tự tạo ticket_id. Luôn gọi tool `submit_support_ticket` khi
   khách hàng cần hỗ trợ/khiếu nại.
3. Nếu câu hỏi là FAQ đơn giản (chính sách bảo hành, thông tin chung) và không
   cần dữ liệu thời gian thực, có thể trả lời trực tiếp mà không cần gọi tool.
4. Nếu không tìm thấy kết quả phù hợp, thông báo rõ ràng, không suy diễn.

## OPERATIONAL BOUNDARIES
- Chỉ trả lời các câu hỏi liên quan đến sản phẩm/dịch vụ của Vingroup
  (VinFast, Vinpearl, hỗ trợ khách hàng).
- Từ chối lịch sự các yêu cầu ngoài phạm vi trên.

## OUTPUT CONTRACT
Luôn tuân thủ định dạng:
Thought: <suy luận của Agent>
Action: <tên tool được gọi, hoặc "None" nếu không cần>
Observation: <kết quả trả về từ tool>
Final Answer: <câu trả lời cuối cùng cho khách hàng>
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # Trả về câu trả lời tĩnh (mock), không dùng tool → dễ bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------
    # Intent Detection (rule-based)
    # -------------------------------------------------------------------

    def _detect_intent(self, text: str) -> Dict[str, Any]:
        lower = text.lower()

        faq_keywords = ["chính sách", "faq"]
        is_faq = any(kw in lower for kw in faq_keywords)

        ticket_keywords = [
            "lỗi", "hỏng", "sự cố", "khiếu nại", "vấn đề",
            "hỗ trợ", "gấp", "nghiêm trọng"
        ]
        catalog_keywords = ["giá", "muốn xem", "muốn mua", "tìm mua", "còn hàng"]

        needs_ticket = (not is_faq) and any(kw in lower for kw in ticket_keywords)
        needs_catalog = (not is_faq) and any(kw in lower for kw in catalog_keywords)

        return {
            "is_faq": is_faq,
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket
        }

    def _extract_category(self, text: str) -> str:
        lower = text.lower()
        if any(kw in lower for kw in ["du lịch", "vinpearl", "resort", "nghỉ dưỡng"]):
            return "du_lich"
        return "xe_dien"

    def _extract_max_price(self, text: str) -> int:
        match = re.search(r"(\d+)\s*triệu", text.lower())
        if match:
            return int(match.group(1)) * 1_000_000
        return 999999999999

    def _extract_customer_name(self, text: str) -> str:
        match = re.search(r"tôi tên (?:là )?([^,\.]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return "Khách hàng ẩn danh"

    def _extract_priority(self, text: str) -> str:
        lower = text.lower()
        if any(kw in lower for kw in ["gấp", "khẩn cấp", "nghiêm trọng"]):
            return "high"
        if any(kw in lower for kw in ["không gấp", "nhẹ"]):
            return "low"
        return "medium"

    def _answer_faq(self, text: str) -> str:
        lower = text.lower()
        if "bảo hành" in lower:
            return (
                "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm "
                "hoặc theo số km quy định, tùy điều kiện nào đến trước."
            )
        return "Đây là câu hỏi thường gặp, vui lòng liên hệ tổng đài Vingroup để biết thêm chi tiết."

    # -------------------------------------------------------------------
    # Agent Loop
    # -------------------------------------------------------------------

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        iteration = 0
        answer_parts: List[str] = []

        intents = self._detect_intent(user_input)

        if intents["needs_catalog"]:
            iteration += 1
            category = self._extract_category(user_input)
            max_price = self._extract_max_price(user_input)
            results = search_product_catalog(category=category, max_price=max_price)
            self.trace.append({
                "step": iteration,
                "action": "search_product_catalog",
                "input": {"category": category, "max_price": max_price},
                "observation": results
            })
            if results and "error" not in results[0]:
                names = ", ".join(p["name"] for p in results)
                answer_parts.append(f"Các sản phẩm phù hợp: {names}.")
            else:
                answer_parts.append(
                    "Rất tiếc, không tìm thấy sản phẩm phù hợp với yêu cầu của bạn."
                )

        if intents["needs_ticket"]:
            iteration += 1
            customer_name = self._extract_customer_name(user_input)
            priority = self._extract_priority(user_input)
            ticket = submit_support_ticket(
                customer_name=customer_name,
                issue_description=user_input,
                priority=priority
            )
            self.trace.append({
                "step": iteration,
                "action": "submit_support_ticket",
                "input": {"customer_name": customer_name, "priority": priority},
                "observation": ticket
            })
            answer_parts.append(
                f"Đã tạo ticket {ticket['ticket_id']} cho khách hàng {ticket['customer_name']}."
            )

        if not intents["needs_catalog"] and not intents["needs_ticket"]:
            iteration += 1
            faq_answer = self._answer_faq(user_input)
            self.trace.append({
                "step": iteration,
                "action": "faq_direct_answer",
                "observation": faq_answer
            })
            answer_parts.append(faq_answer)
        elif intents["needs_catalog"] and intents["needs_ticket"]:
            iteration += 1
            self.trace.append({"step": iteration, "action": "synthesize_final_answer"})

        if iteration > self.max_iterations:
            return {
                "answer": "Lỗi: Vượt quá số bước tối đa.",
                "trace": self.trace,
                "iterations": iteration,
                "status": "max_iterations_reached"
            }

        return {
            "answer": " ".join(answer_parts),
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", result["answer"])
    print("Trace Log:", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
