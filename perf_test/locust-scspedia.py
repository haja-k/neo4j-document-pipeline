from locust import HttpUser, task, events, between
import json
import gevent
import time
import uuid
import requests
from datetime import datetime, timezone
import pandas as pd
import random

# ========================================
# CONFIG
# ========================================

API_URL = "https://i-scs.sarawak.gov.my/v1/chat-messages"
API_KEY = "app-2coFsitSt2Up0L5X4ViZe06V" #hybrid model


MAX_USERS = 100
FAIL_THRESHOLD = 1  # percent fail stop

all_results = {}
summary_rows = []

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

# ========================================
# LOCUST USER CLASS
# ========================================
class ChatUser(HttpUser):
    host = "https://i-scs.sarawak.gov.my/"
    wait_time = between(0, 0)

    @task
    def send_chat(self):
        """Send SSE streaming request and capture detailed metrics."""
        question = random.choice(QUESTIONS)

        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Accept": "text/event-stream",
        }
        payload = {
            "inputs": {},
            "query": question,
            "response_mode": "streaming",
            "conversation_id": "",
            "user": f"loadtest-{uuid.uuid4()}",
        }

        user_id = payload["user"]
        start_time = time.time()

        record = {
            # use timezone-aware UTC timestamp to avoid deprecation warnings
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
            "user_id": user_id,
            "conversation_id": "",
            "request_id": str(uuid.uuid4()),
            "context_type": "chat_message",
            "request": question,
            "response": "",
            "latency_ms": None,
            "elapsed_time_ms": None,
            "ttft_ms": None,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "status": "failed",
            "fail_reason": ""
        }

        last_event = None

        try:
            with requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=400) as resp:
                if resp.status_code != 200:
                    record["fail_reason"] = f"HTTP {resp.status_code}"
                    return record

                ttft = None
                for line in resp.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_name = data.get("event")
                    if event_name:
                        last_event = event_name  # Track last event received

                    # First token time
                    if not ttft:
                        ttft = (time.time() - start_time) * 1000  # ms
                        record["ttft_ms"] = round(ttft, 2)

                    # Capture streaming content
                    if event_name == "message":
                        record["response"] += data.get("text", "")

                    # ========================================
                    # Capture elapsed time from workflow_finished
                    # ========================================
                    elif event_name == "workflow_finished":
                        workflow_data = data.get("data", {})
                        elapsed = workflow_data.get("elapsed_time")

                        if elapsed is not None:
                            record["elapsed_time_ms"] = round(elapsed * 1000, 2)
                        else:
                            created = workflow_data.get("created_at")
                            finished = workflow_data.get("finished_at")
                            if created and finished:
                                record["elapsed_time_ms"] = round((finished - created) * 1000, 2)

                    # ========================================
                    # Capture latency from message_end
                    # ========================================
                    elif event_name == "message_end":
                        meta = data.get("metadata", {})
                        usage = meta.get("usage", {})
                        record["conversation_id"] = data.get("conversation_id", "")
                        record["latency_ms"] = round(float(usage.get("latency", 0)) * 1000, 2)
                        record["input_tokens"] = usage.get("prompt_tokens")
                        record["output_tokens"] = usage.get("completion_tokens")
                        record["total_tokens"] = usage.get("total_tokens")
                        record["status"] = "success"
                        break

                # If no success
                if record["status"] != "success":
                    record["fail_reason"] = f"stopped_at:{last_event or 'no_event'}"

        except Exception as e:
            record["fail_reason"] = f"{str(e)} | last_event:{last_event or 'none'}"
            record["status"] = "failed"

        return record


# ========================================
# CONTROL LOGIC
# ========================================
@events.init.add_listener
def on_locust_init(environment, **_kwargs):
    if not environment.web_ui:
        gevent.spawn(run_incremental_test, environment)


def run_incremental_test(environment):
    print("\n🚀 Starting Sequential SSE Load Test (1→2→3→...→MAX_USERS)\n")

    #incremental test
    for current_users in range(40, MAX_USERS + 1):
        print(f"\n🧩 Step {current_users}: Running {current_users} concurrent users...")
        user_greenlets = []
        step_results = []

    # #direct number of test
    # for current_users in [50]:
    #     print(f"\n🧩 Step {current_users}: Running {current_users} concurrent users...")
    #     user_greenlets = []
    #     step_results = []

        for _ in range(current_users):
            user = ChatUser(environment)
            g = gevent.spawn(run_user_request, user, step_results)
            user_greenlets.append(g)

        gevent.joinall(user_greenlets)

        total = len(step_results)
        failed = len([r for r in step_results if r["status"] == "failed"])
        fail_rate = (failed / total * 100) if total else 0

        all_results[f"Step_{current_users}"] = step_results
        avg_latency = (
            sum(r["latency_ms"] for r in step_results if r["latency_ms"]) /
            max(1, len([r for r in step_results if r["latency_ms"]]))
        )

        summary_rows.append({
            "Step": current_users,
            "Total Requests": total,
            "Failed": failed,
            "Fail Rate (%)": round(fail_rate, 2),
            "Avg Latency (ms)": round(avg_latency, 2),
        })

        print(f"📊 Step {current_users}: {total} total, {failed} failed, Fail rate {fail_rate:.2f}%")

        if fail_rate >= FAIL_THRESHOLD:
            print(f"\n🛑 Fail rate exceeded {FAIL_THRESHOLD}%. Stopping early.")
            break

    print("\n✅ Test finished or stopped.")
    save_results(all_results, summary_rows)
    environment.runner.quit()
    environment.runner.greenlet.kill()


def run_user_request(user, step_results):
    record = user.send_chat()
    step_results.append(record)
    if record["status"] == "failed":
        print(f"❌ {record['user_id']} failed: {record['fail_reason']}")
    else:
        print(
            f"✅ {record['user_id']} success | latency={record['latency_ms']}ms | "
            f"elapsed={record['elapsed_time_ms']}ms"
        )


# ========================================
# SAVE RESULTS TO EXCEL (1 sheet per step + summary)
# ========================================
def save_results(all_results, summary_rows):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scspedia_locust_sse_detailed_results_{timestamp}.xlsx"

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for step, records in all_results.items():
            df = pd.DataFrame(records)
            df.to_excel(writer, sheet_name=step, index=False)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"\n📘 Results saved to {filename}")
    print(f"📈 Summary includes {len(summary_rows)} steps.\n")
