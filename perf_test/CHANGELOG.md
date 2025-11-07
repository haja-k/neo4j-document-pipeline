# Changelog - Enhanced Performance Test

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
