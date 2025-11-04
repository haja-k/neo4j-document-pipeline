from locust import HttpUser, task, events, between
import json
import gevent
from gevent.event import Event
import time
import uuid
import requests
from datetime import datetime, timezone
import pandas as pd
import random
import sys

# Force UTF-8 encoding for console output
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

# ========================================
# CONFIG
# ========================================

API_URL = " "
API_KEY = " " #hybrid model

MAX_USERS = 100
FAIL_THRESHOLD = 1  # percent fail stop
MAX_RESPONSE_TIME = 30000  # maximum acceptable response time in milliseconds

# Global state management
all_results = {}
summary_rows = []
test_stop_event = Event()  # Global event to signal test stop
active_users = 0  # Track current active users

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
    
    def on_start(self):
        """Called when a user starts."""
        global active_users
        if active_users >= MAX_USERS:
            print(f" Maximum users ({MAX_USERS}) reached. Stopping test.")
            test_stop_event.set()
            self.environment.runner.quit()
            return False
        active_users += 1
            
    def on_stop(self):
        """Called when a user stops."""
        global active_users
        active_users -= 1
        print(f"[USER] User stopped. Active users: {active_users}")

    @task
    def send_chat(self):
        """Send SSE streaming request and capture detailed metrics."""
        if test_stop_event.is_set():
            self.environment.runner.quit()
            return
            
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
            "fail_reason": "",
            "response_time": None
        }

        last_event = None

        try:
            with requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=400) as resp:
                if resp.status_code != 200:
                    record["fail_reason"] = f"HTTP {resp.status_code}"
                    record["response_time"] = (time.time() - start_time) * 1000
                    events.request.fire(
                        request_type="SSE",
                        name="chat",
                        response_time=record["response_time"],
                        response_length=0,
                        exception=Exception(f"HTTP {resp.status_code}")
                    )
                    return record

                ttft = None
                for line in resp.iter_lines(decode_unicode=True):
                    if test_stop_event.is_set():
                        record["fail_reason"] = "Test stopped"
                        break
                        
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    event_name = data.get("event")
                    if event_name:
                        last_event = event_name

                    # First token time
                    if not ttft:
                        ttft = (time.time() - start_time) * 1000
                        record["ttft_ms"] = round(ttft, 2)
                        if ttft > MAX_RESPONSE_TIME:
                            test_stop_event.set()
                            record["fail_reason"] = f"TTFT exceeded {MAX_RESPONSE_TIME}ms"
                            break

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
    print("\n Starting Sequential SSE Load Test (1-2-3-...->MAX_USERS)\n")
    
    test_stop_event.clear()  # Reset stop event at start
    global active_users
    active_users = 0  # Reset active users count at start
    
    try:
        for current_users in range(6, MAX_USERS + 1):
            if test_stop_event.is_set() or active_users >= MAX_USERS:
                print("\n Test stopped due to failure conditions or max users reached.")
                print(f"Current active users: {active_users}")
                break
                
            print(f"\n Step {current_users}: Running {current_users} concurrent users...")
            user_greenlets = []
            step_results = []
            
            # Spawn users
            for _ in range(current_users):
                if test_stop_event.is_set():
                    break
                user = ChatUser(environment)
                g = gevent.spawn(run_user_request, user, step_results)
                user_greenlets.append(g)
                
            # Wait for all users to complete or test to stop
            gevent.joinall(user_greenlets, timeout=10)
            
            # Kill any remaining greenlets if test was stopped
            if test_stop_event.is_set():
                for g in user_greenlets:
                    if not g.dead:
                        g.kill()
                        
            total = len(step_results)
            failed = len([r for r in step_results if r["status"] == "failed"])
            fail_rate = (failed / total * 100) if total else 0
            
            # Calculate metrics
            all_results[f"Step_{current_users}"] = step_results
            successful_results = [r for r in step_results if r["latency_ms"]]
            
            if successful_results:
                avg_latency = sum(r["latency_ms"] for r in successful_results) / len(successful_results)
            else:
                avg_latency = 0
                
            summary_rows.append({
                "Step": current_users,
                "Total Requests": total,
                "Failed": failed,
                "Fail Rate (%)": round(fail_rate, 2),
                "Avg Latency (ms)": round(avg_latency, 2),
            })
            
            print(f" Step {current_users}: {total} total, {failed} failed, Fail rate {fail_rate:.2f}%")
            
            # Check failure conditions
            if fail_rate >= FAIL_THRESHOLD:
                print(f"\n Fail rate exceeded {FAIL_THRESHOLD}%. Stopping test.")
                test_stop_event.set()
                break
                
            if avg_latency > MAX_RESPONSE_TIME:
                print(f"\n Average response time ({avg_latency:.2f}ms) exceeded threshold ({MAX_RESPONSE_TIME}ms). Stopping test.")
                test_stop_event.set()
                break
                
    except KeyboardInterrupt:
        print("\n Test interrupted by user.")
        test_stop_event.set()
    except Exception as e:
        print(f"\n Test error: {str(e)}")
        test_stop_event.set()
    finally:
        # Ensure cleanup happens
        test_stop_event.set()

    print("\n Test finished or stopped.")
    save_results(all_results, summary_rows)
    environment.runner.quit()
    environment.runner.greenlet.kill()


def run_user_request(user, step_results):
    try:
        if test_stop_event.is_set() or active_users >= MAX_USERS:
            return
            
        record = user.send_chat()
        if record:  # Only process if we got a record back
            step_results.append(record)
            
            if record["status"] == "failed":
                print(f" {record['user_id']} failed: {record['fail_reason']}")
                events.request.fire(
                    request_type="SSE",
                    name="chat",
                    response_time=record.get("response_time", 0),
                    response_length=0,
                    exception=Exception(record["fail_reason"])
                )
            else:
                print(
                    f" {record['user_id']} success | latency={record['latency_ms']}ms | "
                    f"elapsed={record['elapsed_time_ms']}ms"
                )
                events.request.fire(
                    request_type="SSE",
                    name="chat",
                    response_time=record["latency_ms"],
                    response_length=len(record["response"]),
                    exception=None
                )
            
    except Exception as e:
        print(f" Unexpected error in user request: {str(e)}")
        if not test_stop_event.is_set():
            test_stop_event.set()


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

    print(f"\n Results saved to {filename}")
    print(f" Summary includes {len(summary_rows)} steps.\n")
