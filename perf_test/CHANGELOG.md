# Changelog - Performance Testing Tools

## [3.2.0] - 2025-11-26

### 🔧 Neo4j Phase Profiler - Enhanced Metrics & Configuration

**Updates:** Improved throughput tracking and test configuration flexibility

#### New Features
- **Request Throughput Tracking:** Real wall-clock requests per second (req/sec) measurement
  - Displays actual request completion rate during load tests
  - Separate from theoretical throughput calculations
  - Helps identify true system capacity under concurrent load

- **Embedding Request Metrics:** Track concurrent embedding load
  - Records number of concurrent embedding requests per step
  - Calculates embeddings per request (currently 1:1 ratio)
  - Useful for understanding embedding service pressure

- **Enhanced Error Reporting:** Detailed failure information
  - Shows number of failed vs total requests on failure
  - Displays error messages and status codes for each failed request
  - Helps identify failure patterns during load testing

- **Answer Field Extraction:** Captures full API response answer
  - Stores generated answer text in results
  - Enables response quality analysis alongside performance metrics

#### Configuration Updates
- **Timeout Configuration:** Changed from fixed 120s to `None` (wait indefinitely)
  - Prevents premature test termination during high load
  - Better for stress testing to breaking point
  - Can be customized based on test requirements

- **Load Test Range:** Configurable starting point
  - Changed `MIN_USERS` from 1 to 40 (default, adjustable)
  - Supports jumping to specific load levels (e.g., 80 users)
  - `MAX_USERS` remains at 100
  - `USERS_INCREMENT` set to 4-5 users per step

#### Display Improvements
- **Request Throughput Column:** Added to summary tables
  - Shows "Req/s" in cyan for easy identification
  - Displayed alongside traditional metrics
  - Helps compare actual vs theoretical performance

- **Console Output:** Enhanced step results display
  - Real-time request throughput: "📊 Request Throughput: X.XX req/sec"
  - Shows average embedding time with concurrent user count
  - More informative failure messages with counts

#### Internal Improvements
- Step duration tracking for accurate throughput calculation
- Wall-clock time measurement for request completion rates
- Better separation of derived vs measured metrics

#### When to Use These Features
- **Request Throughput:** Measuring actual API capacity in production-like scenarios
- **Embedding Metrics:** Diagnosing embedding service bottlenecks
- **Enhanced Errors:** Troubleshooting partial failures at scale
- **Timeout Changes:** Stress testing to absolute breaking point

---

## [3.0.0] - 2025-11-18

### 🔍 Neo4j Phase Profiler (NEW)

**New Tool:** Detailed timing analysis for each phase of Neo4j GraphRAG operations

#### What It Measures
- **Embedding Phase:** Vector generation time (200-500ms typical)
- **Hybrid Search Phase:** Neo4j vector + fulltext search (250-500ms typical)
- **MMR Diversification:** Result diversification (20-100ms typical)
- **Cross-Document Coverage:** Multi-document selection (50-150ms typical)
- **Graph Traversal Phase:** Neo4j neighborhood expansion (300-800ms typical)
- **Format Context Phase:** Text formatting (100-300ms typical)

#### Derived Metrics
- **Neo4j Read Time** = hybrid_search + graph_traverse
- **Total Processing Time** = Sum of all phases
- **Network Overhead** = total_request - processing

#### Key Features
- Identifies bottleneck phases automatically
- Tracks performance degradation under load
- Per-phase P95 latency tracking
- Excel reports with phase breakdown
- Optimization recommendations

#### Files
- `neo4j_phase_profiler.py` - Main profiler script
- `PHASE_PROFILER_README.md` - Usage guide
- `PHASES_EXPLAINED.md` - Detailed phase documentation
- `run_phase_profiler.ps1` - Windows quick start
- `run_phase_profiler.sh` - Linux/Mac quick start

#### Usage
```bash
python neo4j_phase_profiler.py
```

#### Example Output
```
Phase Timing Breakdown
┌──────────────────┬──────────┬──────────┬──────────┐
│ Phase            │ Avg (ms) │ P95 (ms) │ % Total  │
├──────────────────┼──────────┼──────────┼──────────┤
│ Embedding        │  450.2   │  520.1   │  25.3%   │
│ Hybrid Search    │  280.5   │  310.2   │  15.8%   │
│ Graph Traverse   │  650.8   │  720.5   │  36.6%   │  ← BOTTLENECK
│ Neo4j Read Total │  931.3   │  980.7   │  52.4%   │
└──────────────────┴──────────┴──────────┴──────────┘
```

#### When to Use
- Understanding where time is spent in queries
- Identifying performance bottlenecks
- Before/after optimization comparisons
- Validating Neo4j index performance
- Checking embedding service health

---

## [2.1.0] - 2025-11-07

### 🎯 Breaking Point Detection

**New Feature:** Automatically find the maximum concurrent users API can handle before failure

#### Detection Logic
- **Failure Threshold:** Stop if fail rate exceeds 50% for 2 consecutive steps
- **Connection Death:** Stop if 80%+ requests have connection/timeout errors
- **Catastrophic Failure:** Stop if fail rate exceeds 80% in single step

#### Configuration
```python
MIN_USERS = 10               # Start at 10 users
MAX_USERS = 200              # Test up to 200 (stops early if breaking point found)
USERS_INCREMENT = 5          # Jump by 5: 10, 15, 20, 25...
FAILURE_THRESHOLD = 50       # % failure rate to trigger stop
CONSECUTIVE_FAILURES = 2     # Number of high-failure steps before stopping
```

#### Output
- **Console:** Shows breaking point when detected
- **Excel Filename:** Includes breaking point (`_bp50` = broke at 50 users)
- **Statistics Sheet:** Records breaking point found and user count
- **Summary:** Highlights breaking point or max tested capacity

#### Example Output
```
🎯 API Breaking Point: 45 concurrent users
Step 8 (45 users): 15/45 success (33.3%), fail rate: 66.7%

Result: graphrag_performance_test_20251107_143022_bp45.xlsx
```

---

## [2.0.0] - 2025-11-07

### 🔧 Critical Fixes

#### False 100% Success Rate
- **Fixed:** Script now checks BOTH HTTP status AND API's `success` field
- **Before:** HTTP 200 responses always counted as successful, even with `{"success": false}`
- **After:** Properly validates `data.get("success", False)` and categorizes all failure types

#### No Queue Building (Not Truly Concurrent)
- **Fixed:** Added `threading.Barrier` to synchronize all thread starts
- **Before:** ThreadPoolExecutor submitted tasks sequentially - no simultaneous burst
- **After:** All threads wait at barrier, then fire at exactly the same time
- **Result:** True concurrent load that actually pressures the queue system

### ✨ New Features

#### Error Categorization
- `success` - Request completed successfully
- `api_failure` - API returned `success: false` (no data available)
- `http_error` - HTTP 4xx/5xx errors
- `timeout` - Request exceeded timeout limit
- `connection_error` - Network/connection issues
- `invalid_json` - Response parsing failed

#### Queue Monitoring
- Shows queue status before and after each load step
- Real-time visibility: `Active=20, Queued=5`
- Validates queue system is working correctly

#### Enhanced Reporting
- Real-time progress with ✓/✗ indicators per request
- Fail rate % prominently displayed in results
- Detailed error breakdown with counts
- Status distribution in final summary
- Color-coded success rates (green >90%, yellow >70%, red <70%)

#### Excel Output Improvements
- **Summary Sheet:** Added fail rate % column and errors breakdown
- **Detailed_Results:** Added status categorization and error messages
- **Step Sheets:** Individual sheets per load level for easy analysis
- **Statistics Sheet:** Overall test performance metrics

### ⚙️ Configuration Changes

**Optimized defaults for queue capacity testing:**
```python
MIN_USERS = 15           # Start near queue limit (20)
MAX_USERS = 30           # Test beyond capacity
USERS_INCREMENT = 5      # Quick progression: 15→20→25→30
WAIT_BETWEEN_STEPS = 3   # Time to observe queue behavior
```

**Test progression:** 15 users → 20 users → 25 users → 30 users
- At 20: Should handle smoothly (at capacity)
- At 25: Should queue 5 requests
- At 30: Should queue 10 requests

### 📊 What to Expect

#### At Queue Capacity (20 users)
```
Queue before: Active=0, Queued=0
Launching 20 threads simultaneously...
Queue after burst: Active=20, Queued=0
Success Rate: ~100%
```

#### Beyond Queue Capacity (25-30 users)
```
Queue before: Active=0, Queued=0
Launching 25 threads simultaneously...
Queue after burst: Active=20, Queued=5  ← Queue working!
Success Rate: ~100% (higher latency for queued requests)
```

### 🚀 Usage

**Quick Test:**
```bash
python enhanced_performance_test.py
```

**Custom Configuration:**
Edit `TestConfig` class in the script:
```python
class TestConfig:
    MIN_USERS = 1
    MAX_USERS = 50
    USERS_INCREMENT = 1
```

**Monitor Queue:**
```powershell
curl https://knowledge-graph.sains.com.my/queue_status
```

### 📦 Dependencies

**Required:**
- requests
- pandas
- openpyxl

**Optional (recommended):**
- rich (for colored console output)

**Install:**
```bash
pip install requests pandas openpyxl rich
```

### 🔍 Validation Checklist

- [ ] At 20 users: Success rate >95%, minimal queueing
- [ ] At 25 users: ~5 requests queued during burst
- [ ] At 30 users: ~10 requests queued during burst
- [ ] Latency increases gradually with queue depth
- [ ] No timeouts or connection errors
- [ ] `api_failure` responses are acceptable (no data available)

### 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Still 100% success | Check Excel `status` column for `api_failure` entries |
| No queue building | Verify `MAX_CONCURRENT_REQUESTS=20` in main.py, test with 50+ users |
| All timeouts | Increase `REQUEST_TIMEOUT`, check API server status |
| Connection errors | Verify API URL, check network/firewall |

---

## [1.0.0] - Initial Release

### Features
- Object-oriented design with separation of concerns
- ThreadPoolExecutor for concurrent requests
- Basic success/failure tracking
- Excel output with multiple sheets
- Configurable test parameters
- Optional Rich library support for better console output

### Metrics
- Success rate calculation
- Average latency
- P95, P99 latencies
- Throughput (requests per second)
- Per-step and overall statistics

### Known Issues
- Not checking API `success` field properly (fixed in 2.0.0)
- Threads not truly concurrent (fixed in 2.0.0)
- No queue monitoring (added in 2.0.0)