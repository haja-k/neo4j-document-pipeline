import requests
import json
import random
import time
import concurrent.futures
from datetime import datetime
import pandas as pd

# ========================================
# CONFIG
# ========================================

API_URL = "https://knowledge-graph.sains.com.my/graphrag"
HEADERS = {
    "Content-Type": "application/json",
}
TIMEOUT = 400  # seconds

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
# FUNCTIONS
# ========================================

def send_request(question):
    """Send a single request and return details."""
    payload = {"question": question}
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=TIMEOUT)
        response_time = (time.time() - start_time) * 1000  # ms
        if response.status_code == 200:
            data = response.json()
            if data.get("success", False):
                return {
                    "timestamp": timestamp,
                    "question": question,
                    "success": True,
                    "response_time": response_time,
                    "response": json.dumps(data),
                    "error": None
                }
            else:
                return {
                    "timestamp": timestamp,
                    "question": question,
                    "success": False,
                    "response_time": response_time,
                    "response": json.dumps(data),
                    "error": "API returned success: false"
                }
        else:
            return {
                "timestamp": timestamp,
                "question": question,
                "success": False,
                "response_time": response_time,
                "response": response.text,
                "error": f"HTTP {response.status_code}"
            }
    except Exception as e:
        response_time = (time.time() - start_time) * 1000
        return {
            "timestamp": timestamp,
            "question": question,
            "success": False,
            "response_time": response_time,
            "response": None,
            "error": str(e)
        }

def test_concurrency(num_concurrent):
    """Test with num_concurrent requests."""
    questions = [random.choice(QUESTIONS) for _ in range(num_concurrent)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = [executor.submit(send_request, q) for q in questions]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # Check if all succeeded
    all_success = all(r["success"] for r in results)
    avg_response_time = sum(r["response_time"] for r in results) / len(results) if results else 0
    return all_success, avg_response_time, results

# ========================================
# SAVE RESULTS TO EXCEL
# ========================================
def save_results(all_results, summary_rows):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"concurrency_i-scs_{timestamp}.xlsx"

    with pd.ExcelWriter(filename, engine="openpyxl") as writer:
        for level, records in all_results.items():
            # records is list of dicts
            df = pd.DataFrame(records)
            df.to_excel(writer, sheet_name=level, index=False)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"\n Results saved to {filename}")
    print(f" Summary includes {len(summary_rows)} concurrency levels.\n")

# ========================================
# MAIN TEST
# ========================================

if __name__ == "__main__":
    print("Starting concurrency test for API...")
    print(f"API URL: {API_URL}")
    print(f"Timeout: {TIMEOUT}s")
    print("-" * 50)

    max_concurrent = 0
    all_results = {}
    summary_rows = []
    
    for num_concurrent in range(1, 101):  # Test up to 100 concurrent
        print(f"Testing {num_concurrent} concurrent requests...")
        success, avg_rt, details = test_concurrency(num_concurrent)
        
        # Store results
        all_results[f"Concurrency_{num_concurrent}"] = details
        
        summary_rows.append({
            "Concurrency Level": num_concurrent,
            "Success": success,
            "Avg Response Time (ms)": round(avg_rt, 2),
            "Total Requests": len(details),
            "Successful Requests": sum(1 for r in details if r["success"]),
            "Failed Requests": sum(1 for r in details if not r["success"])
        })
        
        if success:
            max_concurrent = num_concurrent
            print(f"  SUCCESS: All {num_concurrent} requests succeeded. Avg response time: {avg_rt:.2f}ms")
        else:
            print(f"  FAILED: At least one request failed at {num_concurrent} concurrent requests.")
            print(f"  Details: {details}")
            break
    
    print("-" * 50)
    print(f"Maximum concurrent requests the API can handle: {max_concurrent}")
    print("Test completed.")
    
    # Save to Excel
    save_results(all_results, summary_rows)