import gevent
from gevent import monkey
monkey.patch_all()

from locust import HttpUser, task, between, events
import json
import time
import uuid
import requests
from datetime import datetime, timezone
import random

# ========================================
# CONFIG
# ========================================

API_KEY = " "  # hybrid model
CHAT_ENDPOINT = "/v1/chat-messages"
REQUEST_TIMEOUT = 120  # 2 minutes timeout instead of 400s

# ========================================
# QUESTION LIST
# ========================================
QUESTIONS = [
    "Could you please explain how the mining sector is expected to contribute to inclusivity, employment opportunities, and the development of rural communities in Sarawak?",
    "Can you explain how the handling daily operation of the facility is planned for tourism facilities development?",
    "Can u tell me how state-owned universties help with UP-DLP Sarawak?",
    "Apakah tindakan Jabatan Keselamatan dan Kesihatan Pekerjaan terhadap syarikat yang terlibat dalam bahaya utama di SIP?",
    "Boleh jelaskan berapa projek yang telah dilaksanakan oleh Sr Aman Development Agency (SADA) dan jumlah peruntukan yang diterima untuk tahun 2025?",
    "Can you tell me what are the plans for Totally Protected Areas facilities upgrade? I want know what facilities and timeline for some parks like Matang Wildlife Centre and Kubah National Park.",
    "Could you please explain the main strategies and initiatives outlined for the tourism sector in Sarawak, particularly focusing on how the sector aims to position itself and the key stakeholders involved?",
    "Can you explain how the Rajah Brooke dynasty influenced the cultural and historical development of the Sarawak Delta Geopark, based on its timeline and heritage?",
    "Can you explain the repayment terms for an advance to purchase a new vehicle based on the Sarawak General Order 1996?",
    "Can you provide detailed information on how the projects under the tourism sector ensure timely completion with minimal delays, particularly in relation to the appointment of consultants, contractors, and project monitoring?",
    "Datuk Seri Alexander Nanta Linggi tu dia kerja apa dalam Kabinet Malaysia sekarang? Saya nak tahu betul-betul, dia pegang jawatan apa dan kementerian apa dia uruskan?",
    "Can you explain in detail the strategies and expected outcomes of the Sarawak Heritage Ordinance administration? I want to understand the initiatives, timelines, and resources involved as a Student Researcher studying Sarawak''s development.",
    "What are the key details and benefits of the Sarawak tourism promotion incentives?",
    "Can you provide details on the initiative for Securing Business Events for Miri?",
    "How is Sarawak planning to enhance food production for export, and what role does the Sungai Baji Agropark play in this initiative?",
    "What are the key targets and expected economic impacts of the Business Events 2021 to 2025 initiatives in Sarawak?",
    "Could you please explain how the manufacturing sector's initiatives in the medical & pharmaceutical industry will benefit rural communities in Sarawak, particularly in terms of inclusivity and employment opportunities?",
    "Can you tell me what facilities will be developed at Limbang Mangrove National Park?",
    "How can I obtain permission for use of content from the PCDS 2030 Highlights 2023 document?",
    "Siapakah yang dianggap sebagai exemplary leader dalam profil Toh Puan Fauziah?",
    "Apakah kaitan Toh Puan Fauziah dengan komuniti di Kuala Lumpur?",
    "Who is the State Secretary of Sarawak based on Cabinet Members of Malaysia and Sarawak Government?"
]

class ChatUser(HttpUser):
    host = " " # replace URL
    wait_time = between(1, 3)  # Add small wait between requests per user

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "text/event-stream",
        }

    @task
    def chat_request(self):
        # Prepare request
        payload = {
            "inputs": {},
            "query": self.get_random_question(),
            "response_mode": "streaming",
            "conversation_id": "",
            "user": f"loadtest-{uuid.uuid4()}",
        }

        start_time = time.time()
        events_received = []
        success = False
        exception = None
        response_length = 0

        try:
            # Send request with SSE streaming
            with self.client.post(
                CHAT_ENDPOINT,
                json=payload,
                stream=True,
                timeout=REQUEST_TIMEOUT,
                catch_response=True
            ) as response:
                # Track initial response time
                initial_response_time = time.time() - start_time
                self.environment.events.request.fire(
                    request_type="SSE",
                    name="Initial Response",
                    response_time=initial_response_time * 1000,
                    response_length=0,
                    exception=None,
                    context=payload
                )

                # Process SSE stream
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                        
                    try:
                        data = json.loads(line[5:].strip())
                        event_name = data.get("event")
                        events_received.append(event_name)

                        if event_name == "message":
                            response_length += len(data.get("text", ""))
                        elif event_name == "message_end":
                            success = True
                            break

                    except json.JSONDecodeError:
                        continue

                # Mark final success/failure
                total_time = (time.time() - start_time) * 1000
                if success:
                    response.success()
                    self.environment.events.request.fire(
                        request_type="SSE",
                        name="Complete Stream",
                        response_time=total_time,
                        response_length=response_length,
                        exception=None,
                        context={"events": events_received}
                    )
                else:
                    response.failure(f"Incomplete stream: {','.join(events_received)}")

        except Exception as e:
            exception = e
            self.environment.events.request.fire(
                request_type="SSE",
                name="Failed Request",
                response_time=(time.time() - start_time) * 1000,
                response_length=0,
                exception=e,
                context=payload
            )

    def get_random_question(self):
        return random.choice(QUESTIONS)
